<div align="center">

# Paper Replications and Experiments

**From-scratch PyTorch replications of classic and SOTA AI/ML papers, small language models trained end to end, and post-training experiments**

[![license: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) [![dataset: CC BY 4.0](https://img.shields.io/badge/dataset-CC_BY_4.0-lightgrey)](LICENSE-DATASET) [![Sponsor: GitHub](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/YuvrajSingh-mist) [![Buy me a coffee](https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=black)](https://ko-fi.com/O7W120DR8R)

</div>

I build models from the papers, then train them. Every folder here is code I wrote and ran rather than a fork of someone else's, so the notes next to it record what actually happened: hyperparameters, datasets, final losses, and the parts where my numbers disagree with the paper's.

## What's in here

| Folder | Contents |
|--------|----------|
| [`Replications/`](Replications/) | **37** papers reimplemented from scratch, each self-contained with a README of hyperparameters and results |
| [`SmolHub/`](SmolHub/) | **6** original small language models, 88M–200M parameters, with full training pipelines and HF Space deployments |
| [`Post-Training/`](Post-Training/) | SFT and GRPO recipes, preference optimisation, plus my own runs on my own data |

### [Replications](Replications/)

One folder per paper, from Bahdanau attention and the original Transformer through BERT, GPT, Llama, Mixtral and DeepSeek V3, plus GANs, VAEs, CLIP-style vision-language models and Whisper.

[Browse the replications →](Replications/README.md)

### [SmolHub](SmolHub/)

Small models built and trained end to end on hardware you can rent by the hour:

| Model | Params | Architecture | Dataset |
|-------|--------|--------------|---------|
| [SmolLlama](SmolHub/SmolLlama/) | ~88M | Llama-style, RoPE + GQA | TinyStories |
| [SmolMixtral](SmolHub/SmolMixtral/) | ~124M | Mixture of Experts, 8 experts | TinyStories |
| [SmolTransformer](SmolHub/SmolTransformer/) | ~150M | Classic transformer + RMSNorm | FineWeb |
| [StoryLlama](SmolHub/StoryLlama/) | ~88M | Llama variant, story-tuned | TinyStories |
| [StoryMixtral](SmolHub/StoryMixtral/) | ~200M | Mixtral variant | Story corpus |
| [StoryKimi](SmolHub/StoryKimi/) | ~180M | DeepSeek V3-inspired MLA + MoE | Story corpus |

These are original architectures rather than reproductions. Learned weights live on the Hub — for example [StoryKimi](https://huggingface.co/YuvrajSingh9886/StoryKimi) and [SmolMixtral](https://huggingface.co/YuvrajSingh9886/SmolMixtral) — and the deployable Spaces are under [Spaces](https://huggingface.co/YuvrajSingh9886).

[Browse the small models →](SmolHub/README.md)

### [Post-Training](Post-Training/)

Fine-tuning and post-training work, kept separate from the paper reproductions so the two don't get confused:

- `sft/` — supervised fine-tuning recipes
- `grpo/` — GRPO and RL-style post-training recipes
- `runs/` — my own experiments on my own data, not reproductions of anyone's results

[Browse the post-training experiments →](Post-Training/README.md)

## Deployed models

[smolhub.com](https://smolhub.com) hosts the deployed models, and the Spaces themselves live on the [HuggingFace account](https://huggingface.co/YuvrajSingh9886).

## Usage

Every folder stands alone. Open its README for the hyperparameters and results, then run the notebook or script inside. Datasets marked as local are not committed; the folder README names the source. Weights and checkpoints are never committed — see `.gitignore`.

## Contributing

Feel free to open issues or pull requests if you have suggestions, improvements, or additional replications to add!

## Sponsors

Thank you everyone for their support and love for this project!
- [abstrait](https://github.com/abstrait)
- [Itz-Antaripa](https://github.com/Itz-Antaripa)

### 🔗 Links
[HuggingFace](https://huggingface.co/YuvrajSingh9886) · [Website](https://yuvrajsingh.io) · [Deployed models](https://smolhub.com)

### Authors

- [@YuvrajSingh](https://www.github.com/YuvrajSingh-mist)

## Citation

If you find this repository useful in your research, please cite it:

```bibtex
@software{singh_paper_replications_2026,
  title     = {Paper Replications and Experiments},
  author    = {Singh, Yuvraj},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23198171},
  url       = {https://github.com/YuvrajSingh-mist/Paper-Replications-and-Experiments}
}
```

`10.5281/zenodo.23198171` is the concept DOI: it always resolves to the newest archived version. This release, **v1.1.0** (SmolHub merged in), is pinned at [`10.5281/zenodo.23929573`](https://doi.org/10.5281/zenodo.23929573); the first release, v1.0.0, stays at [`10.5281/zenodo.23198172`](https://doi.org/10.5281/zenodo.23198172).

## License

- **Code and notebooks**: [Apache License 2.0](LICENSE).
- **Documentation, diagrams, generated results and published artifacts**: [CC BY 4.0](LICENSE-DATASET).

Attribution: Yuvraj Singh. See [NOTICE](NOTICE) and [CITATION.cff](CITATION.cff). `SmolHub/` was merged in from a separate repository that declared MIT but shipped no licence file; it is distributed here under Apache-2.0.
