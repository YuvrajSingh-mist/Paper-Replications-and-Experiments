# Replications

One folder per paper, reproduced from scratch in PyTorch. Each folder is self-contained: code, notes, and usually a README with the hyperparameters and losses I actually got.

If you're looking for fine-tuning and post-training runs, those live in [`../Post-Training/`](../Post-Training/).

## Language models and transformers

- **Attention Mechanisms/**: Bahdanau and Luong attention.
- **BERT/**: BERT, pretrained on the Cornell Movie Dialog Corpus.
- **DeepSeekV3/**: DeepSeek model replications.
- **Differential Transformer/**: Differential attention architectures.
- **Encoder-Decoder/**: Encoder-decoder sequence models.
- **Gemma/**, **Gemma3/**: Gemma replications.
- **GPT/**: GPT, trained on TinyShakespeare.
- **Kimi-K2/**: Kimi-K2 replications and training scripts.
- **Llama/**, **Llama4/**: Llama replications.
- **Mixtral/**: Mixture-of-Experts transformers.
- **Moonshine/**: Moonshine model experiments.
- **Transformer/**: The original Transformer and variants.

## Vision and multimodal

- **CLAP/**: Contrastive Language-Audio Pretraining.
- **CLiP/**: CLIP and related vision-language models.
- **Llava/**: Large Language and Vision Assistant.
- **PaliGemma/**: PaliGemma replications.
- **SigLip/**: Sigmoid loss for language-image pretraining.
- **ViT/**: Vision Transformers.

## Generative models

- **CGANs/**: Conditional GANs.
- **CycleGANs/**: Cycle-consistent GANs for image translation.
- **DCGANs/**: Deep Convolutional GANs.
- **Pix2Pix/**: Image-to-image translation.
- **VAE/**: Variational Autoencoders.
- **WGANs/**: Wasserstein GANs.

## Recurrent and sequence models

- **GRU/**: Gated Recurrent Units.
- **LSTM/**: Long Short-Term Memory models.
- **RNNs/**: Recurrent Neural Networks.
- **Seq2Seq/**: Sequence-to-sequence models.

## Preference optimisation and fine-tuning

- **DPO/**: Direct Preference Optimization, trained on Qwen0.5B-Instruct.
- **LoRA/**: Low-Rank Adaptation.
- **ORPO/**: Odds-ratio preference optimization, trained on OPT-330M.
- **SimplePO/**: Simple Preference Optimization.

## Audio and speech

- **CLAP/**: Contrastive Language-Audio Pretraining.
- **TTS/**: Text-to-Speech models.
- **Whisper/**: Whisper speech recognition.

## Training infrastructure

- **DDP/**: Distributed Data Parallel training experiments.

## Usage

Each folder stands alone. Open its README for hyperparameters, datasets and losses, then run the notebook or script inside. Datasets marked as local are not committed; the folder README names the source.
