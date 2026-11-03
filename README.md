<div align="center">

# Paper Replications and Experiments

**From-scratch PyTorch replications of classic and SOTA AI/ML papers, plus my own post-training experiments**

[![license: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) [![dataset: CC BY 4.0](https://img.shields.io/badge/dataset-CC_BY_4.0-lightgrey)](LICENSE-DATASET) [![Sponsor: GitHub](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/YuvrajSingh-mist) [![Buy me a coffee](https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=black)](https://ko-fi.com/O7W120DR8R)

</div>

I build models from the papers, then train them. Every folder here is code I wrote and ran rather than a fork of someone else's, so the notes next to it record what actually happened: hyperparameters, datasets, final losses, and the parts where my numbers disagree with the paper's.

The repository is split in two.

## [Replications](Replications/)

One folder per paper, reproduced from scratch: 37 of them, from Bahdanau attention and the original Transformer through BERT, GPT, Llama, Mixtral and DeepSeek V3, plus GANs, VAEs, CLIP-style vision-language models and Whisper. Each folder is self-contained with a README listing the hyperparameters and results.

[Browse the replications →](Replications/README.md)

## [Post-Training](Post-Training/)

Fine-tuning and post-training work: SFT and GRPO recipes, preference optimisation, and my own runs on my own data.

[Browse the post-training experiments →](Post-Training/README.md)

## Deployed models

Visit [smolhub](https://smolhub.com) to view the deployed models.

## Usage

Every folder stands alone. Open its README for the hyperparameters and results, then run the notebook or script inside. Datasets marked as local are not committed; the folder README names the source.

## Contributing

Feel free to open issues or pull requests if you have suggestions, improvements, or additional replications to add!

## Sponsors

Thank you everyone for their support and love for this project!
- [abstrait](https://github.com/abstrait)
- [Itz-Antaripa](https://github.com/Itz-Antaripa)

### 🔗 Links
[HuggingFace Account](https://huggingface.co/YuvrajSingh9886) · [Website](https://yuvrajsingh.io)

### Authors

- [@YuvrajSingh](https://www.github.com/YuvrajSingh-mist)

## Citation

If you find this repository useful in your research, please cite it:

```bibtex
@software{singh_paper_replications_2025,
  title     = {Paper Replications and Experiments},
  author    = {Singh, Yuvraj},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23198171},
  url       = {https://github.com/YuvrajSingh-mist/Paper-Replications-and-Experiments}
}
```

## License

- **Code and notebooks**: [Apache License 2.0](LICENSE).
- **Documentation, diagrams, generated results and published artifacts**: [CC BY 4.0](LICENSE-DATASET).

Attribution: Yuvraj Singh. See [NOTICE](NOTICE) and [CITATION.cff](CITATION.cff).
