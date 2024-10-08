import torch
import random
import numpy as np
import pickle, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class CustomPromptDataset(Dataset):
    """
    Custom dataset class for handling prompt pairs.

    Attributes:
        prompt_pairs (list): List of tuples containing pairs of prompts.
        tokenizer (AutoTokenizer): Tokenizer to encode the prompts.
    """

    def __init__(self, prompt_pairs, tokenizer):
        """
        Initialize the dataset with prompt pairs and a tokenizer.

        Args:
            prompt_pairs (list): List of tuples containing pairs of prompts.
            tokenizer (AutoTokenizer): Tokenizer to encode the prompts.
        """
        self.prompt_pairs = prompt_pairs
        self.tokenizer = tokenizer

    def __len__(self):
        """Return the number of prompt pairs."""
        return len(self.prompt_pairs)

    def __getitem__(self, idx):
        """
        Get a tokenized pair of prompts by index.

        Args:
            idx (int): Index of the prompt pair.

        Returns:
            tuple: Tokenized pair of prompts.
        """
        prompt1, prompt2 = self.prompt_pairs[idx]
        encoding1 = self.tokenizer(prompt1, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        encoding2 = self.tokenizer(prompt2, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        
        return encoding1, encoding2

def custom_collate_fn(batch):
    """
    Custom collate function to handle batch of tokenized prompt pairs.

    Args:
        batch (list): List of tokenized prompt pairs.

    Returns:
        tuple: Batch of tokenized prompt pairs.
    """
    encoding1 = {key: torch.cat([d[key] for d, _ in batch], dim=0) for key in batch[0][0]}
    encoding2 = {key: torch.cat([d[key] for _, d in batch], dim=0) for key in batch[0][1]}
    return encoding1, encoding2

def create_dataloader(prompt_pairs, tokenizer, batch_size=4):
    """
    Create a DataLoader for the prompt pairs.

    Args:
        prompt_pairs (list): List of tuples containing pairs of prompts.
        tokenizer (AutoTokenizer): Tokenizer to encode the prompts.
        batch_size (int, optional): Batch size. Defaults to 4.

    Returns:
        DataLoader: DataLoader for the prompt pairs.
    """
    dataset = CustomPromptDataset(prompt_pairs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    return dataloader

def load_data(data_name, tokenizer):
    """
    Load prompt pairs from a pickle file and create a DataLoader.

    Args:
        data_name (str): Name of the pickle file containing prompt pairs.
        tokenizer (AutoTokenizer): Tokenizer to encode the prompts.

    Returns:
        DataLoader: DataLoader for the prompt pairs.
    """
    with open(data_name, 'rb') as file:
        prompt_pairs = pickle.load(file)
    dataloader = create_dataloader(prompt_pairs, tokenizer, batch_size=4)
    return dataloader

def load_model_and_tokenizer(model_dir=None):
    """
    Load an initial model and tokenizer.

    Args:
        model_dir (str, optional): Directory to load the model from. Defaults to None.

    Returns:
        tuple: Loaded model and tokenizer.
    """
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if(model_dir=="./mistralai/4"):
        chat_template = open('./mistral-instruct.jinja').read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        tokenizer.chat_template = chat_template

    return model, tokenizer

def train_model(model, dataloader, output_dir_name):
    """
    Train the model using LoRA and mixed precision training.

    Args:
        model (AutoModelForCausalLM): Model to be trained.
        dataloader (DataLoader): DataLoader for training data.
        output_dir_name (str): Directory to save the trained model.
    """
    # LoRA configuration
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Type of task
        r=8,  # LoRA rank
        lora_alpha=32,  # Scaling parameter
        lora_dropout=0.2,  # Dropout rate
    )

    # Get the PEFT model
    model = get_peft_model(model, config)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    def kl_divergence_loss(output1, output2):
        """
        Calculate the Kullback-Leibler divergence loss.

        Args:
            output1 (Tensor): First set of logits.
            output2 (Tensor): Second set of logits.

        Returns:
            Tensor: Calculated KL divergence loss.
        """
        return F.kl_div(F.log_softmax(output1, dim=-1), F.softmax(output2, dim=-1), reduction='batchmean')

    # Mixed precision training
    scaler = GradScaler()

    # Fine-tune the model
    model.train()
    num_epochs = 20
    accumulation_steps = 4

    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(dataloader):
            encoding1, encoding2 = batch
            
            input_ids1 = encoding1['input_ids'].to(model.device)
            attention_mask1 = encoding1['attention_mask'].to(model.device)
            input_ids2 = encoding2['input_ids'].to(model.device)
            attention_mask2 = encoding2['attention_mask'].to(model.device)
            
            with autocast():
                outputs1 = model(input_ids=input_ids1, attention_mask=attention_mask1).logits
                outputs2 = model(input_ids=input_ids2, attention_mask=attention_mask2).logits
                
                loss = kl_divergence_loss(outputs1, outputs2)
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
    
    model.save_pretrained('./'+output_dir_name)

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


def main():
    """
    Main function to parse arguments, load data, and train the model.
    """
    parser = argparse.ArgumentParser(description="Fine-tune model on prompt style variants")
    parser.add_argument('--output_dir_name', type=str, required=True, help="Directory to save the model checkpoint")
    parser.add_argument('--training_data_filename', type=str, required=True, help="Name of the training data pickle file")
    parser.add_argument('--seed', type=int, required=True, help="Random Seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(model_dir = './mistralai/4')
    tokenizer.pad_token = tokenizer.eos_token
    dataloader = load_data(args.training_data_filename, tokenizer)
    train_model(model, dataloader, args.output_dir_name)

if __name__ == "__main__":
    main()
