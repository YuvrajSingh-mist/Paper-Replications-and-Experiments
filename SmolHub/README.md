# 🚀 SmolHub: Small Language Models from Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](../LICENSE)

> *Building powerful small language models (100-300M parameters) completely from scratch!*

> **Note:** this was its own repository until it was merged in here. Clone
> `Paper-Replications-and-Experiments` and work inside `SmolHub/`; the old
> `SmolHub` repo is archived and no longer updated.

## 🎯 Overview

SmolHub is a collection of **small but mighty language models** implemented entirely from scratch using PyTorch. This repository focuses on creating efficient, lightweight LLMs that can run on modest hardware while still delivering impressive performance. All models are designed to be in the 100-300M parameter range, making them perfect for research, experimentation, and resource-constrained environments.

## 🏗️ Architecture Implementations

### 📚 Available Models

| Model | Parameters | Architecture | Key Features | Training Dataset |
|-------|------------|--------------|-------------|------------------|
| **SmolLlama** | ~130M | Llama-inspired, instruct-tuned | FineWeb → SFT → DPO pipeline | FineWeb 10BT |
| **SmolMixtral** | ~124M (8x12M) | Mixture of Experts | Sparse activation, Flash Attention, SwiGLU | TinyStories (1M texts, 14K steps) |
| **SmolTransformer** | ~150M | Standard Transformer | Classic attention mechanism, RMSNorm | FineWeb |
| **StoryLlama** | ~88M | Llama-inspired | RoPE, SwiGLU, MQA/GQA, RMSNorm | TinyStories (4B tokens, 5K steps) |
| **StoryMixtral** | ~200M | Mixtral variant | Story-focused MoE training | Custom story dataset |
| **StoryKimi** | ~180M | Custom architecture | Optimized for narrative generation | Story corpus |

### 🏷️ Weights and demos on Hugging Face

Each model's trained weights and its live demo live on the Hugging Face Hub.
`-Zero` Spaces are the ZeroGPU variants.

| Model | Weights | Space |
|-------|---------|-------|
| **SmolLlama** | — | [SmolLlama](https://huggingface.co/spaces/YuvrajSingh9886/SmolLlama) · [SmolLlama-Zero](https://huggingface.co/spaces/YuvrajSingh9886/SmolLlama-Zero) |
| **SmolMixtral** | [SmolMixtral](https://huggingface.co/YuvrajSingh9886/SmolMixtral) | [SmolMixtral-Zero](https://huggingface.co/spaces/YuvrajSingh9886/SmolMixtral-Zero) |
| **SmolTransformer** | [SmolTransformer](https://huggingface.co/YuvrajSingh9886/SmolTransformer) | [SmolTransformer-Zero](https://huggingface.co/spaces/YuvrajSingh9886/SmolTransformer-Zero) |
| **StoryLlama** | [StoryLlama](https://huggingface.co/YuvrajSingh9886/StoryLlama) | [StoryLlama](https://huggingface.co/spaces/YuvrajSingh9886/StoryLlama) · [StoryLlama-Zero](https://huggingface.co/spaces/YuvrajSingh9886/StoryLlama-Zero) |
| **StoryMixtral** | [StoryMixtral](https://huggingface.co/YuvrajSingh9886/StoryMixtral) | [StoryMixtral-Zero](https://huggingface.co/spaces/YuvrajSingh9886/StoryMixtral-Zero) |
| **StoryKimi** | [StoryKimi](https://huggingface.co/YuvrajSingh9886/StoryKimi) | [StoryKimi-Zero](https://huggingface.co/spaces/YuvrajSingh9886/StoryKimi-Zero) |

## 📦 Citation

This directory is archived with the rest of the repository as
**Paper Replications and Experiments**
([concept DOI](https://doi.org/10.5281/zenodo.23198171)). Cite that record if you
use the models, the pipelines or the architectures here.

### 🔧 Key Components Implemented

#### Attention Mechanisms
- **Multi-Head Attention (MHA)**: Classic transformer attention
- **Multi-Query Attention (MQA)**: Shared key-value heads for efficiency  
- **Grouped-Query Attention (GQA)**: Balanced efficiency and performance
- **Flash Attention**: Memory-efficient attention computation

#### Position Encodings
- **Rotary Position Embeddings (RoPE)**: Advanced position encoding
- **Learned Position Embeddings**: Traditional approach

#### Activation Functions & Normalization
- **SwiGLU**: Gated Linear Unit with Swish activation
- **Swish**: Smooth activation function
- **RMSNorm**: Root Mean Square Layer Normalization
- **LayerNorm**: Standard layer normalization

#### Advanced Features
- **Mixture of Experts (MoE)**: 8 experts with top-2 routing in SmolMixtral
- **Noisy Top-K Routing**: Enhanced expert selection
- **Weight Tying**: Shared embedding and output projection weights
- **Gradient Checkpointing**: Memory optimization during training

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision transformers datasets wandb tqdm
```

### Training a Model

```bash
# Navigate to any model directory
cd SmolMixtral

# Install dependencies
bash install.sh

# Start training
python trainer.py
```

### Running Inference

```python
from inference import generate_text

# Load your trained model
model = load_model("path/to/checkpoint")

# Generate text
output = generate_text(
    model=model,
    prompt="Once upon a time",
    max_length=100,
    temperature=0.7
)
print(output)
```

## 📊 Model Performance & Results

### Training Achievements
- **SmolMixtral**: Successfully trained 124M parameter MoE model
  - 📈 [View Training Report on WandB](https://wandb.ai/rentio/Mixtral-DDP-Pretrain-10-billion-tokens/reports/SmolMixtral--VmlldzoxMzYyNzc0OQ?accessToken=nybd4lxybsbq5k5fh2dqjcucdawilt3fossn583wv6jiu8tbdzcybiihe7rhsqmq)
  - 💾 [Pre-trained weights on HuggingFace](https://huggingface.co/YuvrajSingh9886/SmolMixtral)

### Key Performance Features
All models are optimized for:
- **Fast inference** on consumer GPUs (RTX 3090, RTX 4090)
- **Low memory footprint** (fits in 8GB VRAM)
- **Quality text generation** with coherent outputs
- **Efficient training** with gradient checkpointing and mixed precision

### Benchmark Results
- Generated high-quality stories and narratives
- Competitive perplexity scores for model size
- Fast inference speeds (>100 tokens/second on RTX 4090)

## 🏋️ Training Features

### Advanced Training Techniques
- **Distributed Training**: Multi-GPU support with PyTorch DDP
- **Memory Optimization**: 
  - Gradient checkpointing for large models
  - Mixed precision training (FP16/BF16)
  - Flash Attention for memory efficiency
- **Advanced Scheduling**: 
  - Cosine annealing with warmup
  - Learning rate scheduling with restarts
- **Monitoring**: 
  - Wandb integration for experiment tracking
  - Real-time loss visualization
  - Gradient norm monitoring
- **Checkpointing**: 
  - Automatic model saving and resuming
  - State-dict preservation for optimizers and schedulers

### Optimization Features
- **AdamW Optimizer**: With configurable weight decay
- **Gradient Clipping**: Prevents exploding gradients
- **Dynamic Loss Scaling**: For mixed precision training
- **Expert Load Balancing**: For MoE models (auxiliary loss)
- **Tokenizer Integration**: Custom BPE and GPT-2 tokenizers

## 🗂️ Repository Structure

```
SmolHub/
├── SmolMixtral/          # Mixture of Experts implementation
├── SmolTransformer/      # Standard transformer architecture  
├── StoryLlama/          # Llama-inspired model for stories
├── StoryMixtral/        # Mixtral variant for narratives
├── StoryKimi/           # Custom story-focused architecture
├── smolhub_hub/         # Package management utilities
└── README.md            # This file
```

Each model directory contains:
- `model.py` - Core architecture implementation
- `trainer.py` - Training loop and optimization
- `inference.py` - Text generation utilities
- `config.py` - Model configuration
- `tokenizer.py` - Tokenization utilities
- `gradio/` - Interactive web demos

## 🎮 Interactive Demos

Each model comes with a Gradio-powered web interface for easy experimentation:

```bash
cd SmolMixtral/gradio
python app.py
```

## 📈 Training Data & Methodology

### Datasets Used
- **TinyStories**: High-quality short stories for narrative models
  - SmolMixtral: 1M texts, 14K training steps
  - StoryLlama: 4B tokens, 5K training steps
- **FineWeb**: Curated high-quality web text (10BT sample)
- **OpenWebText**: Diverse internet content
- **Custom datasets**: Domain-specific content for specialized models

### Training Methodology
- **From Scratch Training**: All models trained without pre-training transfer
- **Progressive Training**: Gradual increase in sequence length and complexity
- **Data Quality Focus**: Careful dataset curation and filtering
- **Evaluation Strategy**: Regular validation on held-out sets
- **Text Generation**: Real-time quality assessment during training

## 🔬 Research & Experiments

This repository serves as a research platform for:
- **Architecture exploration**: Testing new attention mechanisms
- **Scaling laws**: Understanding parameter efficiency
- **Training techniques**: Experimenting with optimization strategies
- **Evaluation**: Comprehensive benchmarking of small models

## 🤝 Contributing

Contributions are welcome! Whether you want to:
- Add new model architectures
- Improve training efficiency
- Add evaluation scripts
- Fix bugs or improve documentation

Please feel free to open issues and pull requests.

## 📄 License

This project is licensed under the Apache License 2.0, like the rest of this
repository - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the incredible work on Transformer architectures
- Built with PyTorch and the amazing open-source ML community
- Special thanks to papers on MoE, RoPE, and efficient attention mechanisms

## 📞 Contact

- **GitHub**: [@YuvrajSingh-mist](https://github.com/YuvrajSingh-mist)
- **Issues**: Feel free to open GitHub issues for questions or bugs

---

*"Small models, big possibilities!"* 🌟

## 🗺️ Roadmap

- [ ] Add more architecture variants
- [ ] Implement quantization techniques
- [ ] Add comprehensive benchmarks
- [ ] Create model comparison tools
- [ ] Add fine-tuning scripts
- [ ] Implement RLHF training
