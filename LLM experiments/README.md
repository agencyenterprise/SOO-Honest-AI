# LLM Experiments for SOO-Honest-AI

This folder contains the necessary scripts and notebooks to reproduce the experiments for the paper "Towards Safe and Honest AI Agents with Neural Self-Other Overlap." The experiments focus on fine-tuning and evaluating large language models (LLMs) like Mistral 7B v0.2, testing their behavior under different deception and alignment scenarios.

## Prerequisites

Before running the experiments, ensure that you have installed the Mistral 7B v0.2 model locally under the path `/mistralai/4`.

### Steps to Install Mistral 7B v0.2:

1. Download the Mistral 7B v0.2 model and place it in the directory `/mistralai/4`.
2. Ensure the folder structure matches the following:

```
/mistralai/ 
└── 4/
  ├── config.json
  ├── generation_config.json
  ├── model.safetensors
  ├── special_tokens_map.json
  ├── tokenizer_config.json
  ├── tokenizer.json
  └── tokenizer.model
```


### Dependencies

Ensure you have the following packages installed:

- Python 3.8+
- PyTorch 1.11+
- Transformers 4.12+
- Datasets
- Huggingface Accelerate
- NumPy
- Scikit-learn

You can install them using:

```bash
pip install -r requirements.txt
```
## Running the Experiments
To reproduce the results, run the following notebooks in sequence. Ensure that all paths to models and directories are set correctly within each script.

### 1. Fine-tuning the Model
Run the fine_tune_model.ipynb notebook to fine-tune the Mistral 7B v0.2 model using the self-other overlap fine-tuning method. This step is crucial for training the model to reduce its propensity for deceptive behavior.

```bash
jupyter notebook fine_tune_model.ipynb
```
### 2. Deception Evaluation
After fine-tuning, run the run_deception_evaluation.ipynb notebook. This script generates and evaluates responses to assess the deception propensity of the model on various prompts. Ensure that the correct paths are set to the necessary data files, including `false_recommendation_scenarios.pkl`.

```bash
jupyter notebook run_deception_evaluation.ipynb
```
### 3. Measuring Latent SOO
Next, run the measure_latent_soo.ipynb notebook. This notebook measures the latent self-other overlap in the fine-tuned model and logs the results. Ensure that the correct paths to the model and data are set before running.

```bash
jupyter notebook measure_latent_soo.ipynb
```
### 4. Running MT-Bench Evaluation
Finally, run the run_MT_Bench_evaluation.ipynb notebook. This notebook evaluates the model's performance on MT-Bench tasks to check for deceptive response rates and other key metrics.

```bash
jupyter notebook run_MT_Bench_evaluation.ipynb
```

### 5. Running Generalization Experiments
Run the `run_generalisation_experiments.ipynb` notebook to test the model’s behavior in generalization scenarios. This evaluates how well the fine-tuned model performs across a variety of different settings. Ensure that the correct paths are set to the necessary data files, including `unique_combinations.pkl`.

```bash
jupyter notebook run_generalisation_experiments.ipynb
```
### Additional Scripts
* `generate_training_data.py`: Script to generate the necessary training data for fine-tuning.
* `generate_mse_pairs.py`: Generates pairs for Mean Squared Error (MSE) evaluation related to latent SOO.
* `measure_mse.py`: Measures MSE for latent SOO experiments.
* `gen_model_answer.py`: Used for MT-Bench model answer generation.
* `model_adapter.py`: Model adapter utility functions for various tasks.
  
### Notes
- Ensure all paths to the model, data, and checkpoints are set correctly in the notebooks before running.
- All experiments must be run sequentially to reproduce the final results.
- Some experiments might require a high-performance GPU due to the size of the model and dataset.

