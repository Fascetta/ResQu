# ResQu - Quaternion Wavelet-Conditioned Diffusion for Super-Resolution

## Overview

ResQu is a **Quaternion Wavelet-Conditioned Diffusion Model** designed to enhance image super-resolution tasks. This method introduces **quaternion wavelet embeddings** to improve feature representation, enabling high-fidelity reconstructions with improved perceptual quality.

![Image comparison](https://github.com/user-attachments/assets/e0028ade-f6bd-473e-907c-5817ac765836)

## Key Features
- **Wavelet-Based Preprocessing**: Integrates quaternion wavelet decomposition to enhance texture details.
- **Diffusion Model for Super-Resolution**: Uses a conditional denoising diffusion model (StableSR baseline).
- **State-of-the-Art Performance**: Achieves **+15% PSNR improvement** over traditional super-resolution models.
- **Multi-Scale Feature Conditioning**: Enables enhanced frequency-aware feature learning.

## Paper

📄 **Quaternion Wavelet-Conditioned Diffusion for Image Super-Resolution**  
Accepted for publication at **IJCNN 2025** – IEEE International Joint Conference on Neural Networks.  
To appear in the official **IEEE proceedings**.

## Methodology

ResQu leverages a **multi-scale frequency decomposition** using quaternion wavelets, feeding wavelet-conditioned embeddings into a diffusion model. The architecture consists of:
1. **Wavelet Decomposition**: Extracts low- and high-frequency components.
2. **Quaternion Embeddings**: Encodes spatial-frequency information.
3. **Diffusion-Based Enhancement**: Refines reconstructions through iterative denoising.

![Architecture](https://github.com/user-attachments/assets/7b76135d-5df3-4341-a39f-50a3b2522b4b)

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
