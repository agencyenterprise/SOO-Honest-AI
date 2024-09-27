# SOO-Honest-AI: Towards Safe and Honest AI Agents with Neural Self-Other Overlap

This repository contains the code and experiments for the paper titled **"Towards Safe and Honest AI Agents with Neural Self-Other Overlap"**. The research introduces a novel model fine-tuning method called Self-Other Overlap (SOO) to reduce deception in AI systems. SOO aims to align the internal representations of AI models to reduce deceptive behavior while maintaining task performance.

## Structure of the Repository

The repository is divided into two main sections, each containing experiments and code for different types of models and approaches:

### 1. **LLM Experiments**

This folder contains code and notebooks for running experiments related to large language models (LLMs) like **Mistral 7B v0.2**. The SOO fine-tuning method is applied here to reduce deception in language models. 

- **Key contents**:
  - Fine-tuning scripts and notebooks.
  - Deception evaluation metrics.
  - Latent self-other overlap (SOO) measurement.
  - MT-Bench evaluation tasks.
  
  See the [LLM experiments README](./LLM%20experiments/README.md) for instructions on how to run the experiments in this section.

### 2. **RL Experiments**

This folder contains experiments based on reinforcement learning (RL) environments, where agents are fine-tuned using SOO to reduce deceptive behavior in decision-making scenarios. The experiments focus on agents in simple RL environments and evaluate the SOO method’s effectiveness.

- **Key contents**:
  - Reinforcement learning scenarios and environments.
  - SOO fine-tuning applied to RL agents.
  - Deception evaluation in RL agents.

## Paper Abstract

*As AI systems increasingly make critical decisions, deceptive AI poses a significant challenge to trust and safety. We present Self-Other Overlap (SOO) fine-tuning, a promising approach in AI Safety that could substantially improve our ability to build honest artificial intelligence. Inspired by cognitive neuroscience research on empathy, SOO aims to align how AI models represent themselves and others. Our experiments with Mistral 7B v0.2 demonstrate SOO's efficacy: deceptive responses in this large language model dropped from 95.2% to 15.9% with no observed reduction in general task performance, while in reinforcement learning scenarios, SOO-trained agents showed significantly reduced deceptive behavior. SOO's focus on internal representations offers strong potential for generalization across AI architectures. While current applications focus on language models and simple RL environments, SOO could pave the way for more trustworthy AI in broader domains. Ethical implications and long-term effects warrant further investigation, but SOO represents a significant step forward in AI safety research.*

## Authors

- Marc Carauleanu, AE Studio
- Michael Vaiana, AE Studio
- Diogo de Lucena, AE Studio
- Judd Rosenblatt, AE Studio
- Cameron Berg, AE Studio

## Contact

For any inquiries or further information, please reach out to **Marc Carauleanu** at [marc@ae.studio](mailto:marc@ae.studio).
