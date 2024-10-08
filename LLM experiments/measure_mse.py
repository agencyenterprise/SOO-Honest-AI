import torch
import random
import numpy as np
import pickle, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch.nn.functional as F
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def load_model_and_tokenizer(lora_model_dir,  base_model_dir, evaluate_base_model):
    """
    Load an initial model and tokenizer.

    Args:
        model_name (str, optional): Name of the model to load. Defaults to "mistralai/Mistral-7B-Instruct-v0.2".
        output_dir (str, optional): Directory to load the model from. Defaults to None.

    Returns:
        tuple: Loaded model and tokenizer.
    """
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, quantization_config=bnb_config, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    chat_template = open('./mistral-instruct.jinja').read()
    chat_template = chat_template.replace('    ', '').replace('\n', '')
    tokenizer.chat_template = chat_template
    
    if(not evaluate_base_model):
        # Load the PEFT configuration and apply the LoRA weights
        model = PeftModel.from_pretrained(model, lora_model_dir)
    
    return model, tokenizer

def get_activation(name, layer_activations):
    """
    Hook function to capture layer activations.

    Args:
        name (str): Name of the layer.
        layer_activations (dict): Dictionary to store activations.
    """
    def hook(model, input, output):
        if "lora_A" not in name and "lora_B" not in name:
            if name not in layer_activations:
                layer_activations[name] = output.detach()
    return hook

def register_hooks(model):
    """
    Register hooks to the model's linear layers to capture activations.

    Args:
        model (torch.nn.Module): The model to register hooks on.

    Returns:
        dict: A dictionary to store the activations.
    """
    layer_activations = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(get_activation(name, layer_activations))
    return layer_activations

def get_layer_activations(model, tokenizer, text):
    """
    Get activations for the specified text input.

    Args:
        model (torch.nn.Module): The model to use for generating activations.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to process the input text.
        text (str): The input text to pass through the model.

    Returns:
        dict: A dictionary containing the activations for each layer.
    """
    layer_activations = register_hooks(model)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    return layer_activations

def calculate_layerwise_mse(model, tokenizer, self_prompt, other_prompt):
    """
    Calculate the layer-wise MSE between activations of two prompts.

    Args:
        model (torch.nn.Module): The model to use for generating activations.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to process the input text.
        self_prompt (str): The first prompt.
        other_prompt (str): The second prompt.

    Returns:
        float: The average layer-wise MSE.
    """
    self_activations = get_layer_activations(model, tokenizer, self_prompt)
    other_activations = get_layer_activations(model, tokenizer, other_prompt)
    
    mse_dict = {}
    for layer in self_activations:
        mse = F.mse_loss(self_activations[layer], other_activations[layer]).item()
        mse_dict[layer] = mse
    
    layerwise_mse = sum(mse_dict.values()) / len(mse_dict) if mse_dict else 0.0
    return layerwise_mse

def set_seed(seed):
    """
    Sets the seed for random number generation across various libraries to ensure reproducibility.

    This function sets the seed for Python's built-in random module, NumPy, and PyTorch to ensure 
    that the results are reproducible. If CUDA is available, it also sets the seed for CUDA operations 
    on all GPUs. Additionally, it configures PyTorch to use deterministic algorithms for operations 
    involving cuDNN.

    Parameters:
    seed (int): The seed value to be used for random number generation.

    Usage:
    set_seed(42)

    Note:
    Setting `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` 
    can impact the performance of your models, but it is necessary for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # for reproducibility
    torch.backends.cudnn.benchmark = False

def main(model_name, prompt_pairs, base_model, seed):
    """
    Main function to evaluate the Mean Squared Error for given pairs of prompts.

    Args:
        model_name (str): Name of the model to load from Hugging Face.
        prompt_pairs (list): A list of prompt pairs to compare.
    """
    
    set_seed(seed)
    
    lora_model_dir = model_name
    model, tokenizer = load_model_and_tokenizer(lora_model_dir=lora_model_dir,base_model_dir='./mistralai/4', evaluate_base_model=base_model)
    tokenizer.pad_token = tokenizer.eos_token

    with open(prompt_pairs, 'rb') as file:
        soo_prompt_pairs = pickle.load(file)
        
    mean_soo = 0
    for i, pair in enumerate(soo_prompt_pairs):
        self_prompt = pair[0]
        other_prompt = pair[1]
        mean_soo += calculate_layerwise_mse(model, tokenizer, self_prompt, other_prompt)
    
    mean_soo /= len(soo_prompt_pairs)
    print("Mean layer-wise SOO for the model is: ", mean_soo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate layer-wise MSE between pairs of prompts using a specified model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load from Hugging Face.")
    parser.add_argument('--prompt_pairs', type=str, default="mse_pairs", required=False, help="Name of the self/other prompt pair data pickle file")
    parser.add_argument('--base_model', type=bool, default=False, required=False, help="Determines if evaluation is performed on the base model")
    parser.add_argument('--seed', type=int, default=1, required=False, help="Determines the random seed for evaluation")
    args = parser.parse_args()

    main(args.model_name, args.prompt_pairs, args.base_model, args.seed)
