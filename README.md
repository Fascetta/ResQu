# Super-Resolution with Quave Preprocessing and StableSR Framework

<p align="center">
  <img src="https://user-images.githubusercontent.com/22350795/236680126-0b1cdd62-d6fc-4620-b998-75ed6c31bf6f.png" height="60">
</p>

## Research Overview

We propose an enhanced approach to real-world image super-resolution by integrating **Quave preprocessing** into the pipeline, enabling richer feature embeddings to feed into the **time-aware encoder** of the StableSR framework.

This work combines the strengths of **StableSR**, a state-of-the-art super-resolution framework, and **QUAVE**, a quaternion wavelet-based preprocessing tool, to achieve superior generalization and performance for image analysis tasks.

### Baseline: StableSR Framework

This work builds on the **StableSR** model, which leverages diffusion priors for real-world image super-resolution.

- **Reference**: [StableSR Paper](https://arxiv.org/abs/2305.07015)
- **Code Repository**: [StableSR GitHub](https://github.com/IceClear/StableSR)

### Key Contributions
- **Quave Preprocessing**: Extracts advanced embeddings to enhance input quality.
- **Time-Aware Encoder**: Integrates temporal and feature-rich embeddings for improved image fidelity.
- **Real-World Applications**: Targets arbitrary upscaling with minimal artifacts.

---

## Pipeline Overview

<img src="assets/network.png" width="800px" alt="Pipeline Overview">

1. **Input Preprocessing**: Quave processes the low-resolution (LR) image, extracting salient sub-band features.
2. **Diffusion Prior**: StableSR’s encoder-decoder generates latent codes.
3. **Time-Aware Encoding**: Combines Quave embeddings with temporal features for enhanced decoding.

---

## Integration of QUAVE

### About Quave
The **Quaternion Wavelet Network (QUAVE)** is a novel framework designed to generalize image representations. It enhances neural model performance by extracting and selecting frequency sub-bands to provide approximation and fine-grained features, offering a more complete input representation for image processing tasks.

- **Reference**: [Quave Paper](https://arxiv.org/abs/2310.10224)
- **Authors**: Luigi Sigillo, Eleonora Grassucci, Aurelio Uncini, Danilo Comminiello
- **Code Repository**: [Quave GitHub](https://github.com/)

---

## Running the Model

### Dependencies
- **Pytorch**: 1.12.1
- **CUDA**: 11.7
- **Quave**
- **Other**: See `environment.yaml`

### Training
Run the training pipeline:
```bash
python main.py --train --base configs/stableSRNew/v2-finetune_text_T_quave.yaml --gpus 0 --name "SuperRes_Quave" --scale_lr False
```

### Testing
Test real-world super-resolution performance:
```bash
python scripts/sr_test_quave.py --config configs/stableSRNew/v2-test_quave.yaml --ckpt ./models/quave_stablesr.ckpt --input ./inputs/test_image.png --output ./outputs/
```

---

## Results

### Real-World Performance
<div align="center">
</div>

- Enhanced detail retention with Quave embeddings.
- Improved temporal coherence in time-aware sequences.

---

## Development Status

This project is currently **under active development**. Some features and components may not be finalized yet. We are also in the process of preparing a **research paper** that documents our findings and contributions in detail.

Stay tuned for updates!

---

## Citations

### StableSR
```bibtex
@article{wang2024exploiting,
  author = {Wang, Jianyi and Yue, Zongsheng and Zhou, Shangchen and Chan, Kelvin C.K. and Loy, Chen Change},
  title = {Exploiting Diffusion Prior for Real-World Image Super-Resolution},
  journal = {International Journal of Computer Vision},
  year = {2024}
}
```

### Quave
```bibtex
@misc{sigillo2024generalizing,
      title={Generalizing Medical Image Representations via Quaternion Wavelet Networks}, 
      author={Luigi Sigillo and Eleonora Grassucci and Aurelio Uncini and Danilo Comminiello},
      year={2024},
      eprint={2310.10224},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

---

## Acknowledgment

This project is based on the StableSR framework developed by researchers at Nanyang Technological University and the QUAVE framework developed at Sapienza University. Their combined capabilities offer a powerful approach to image super-resolution.

---
