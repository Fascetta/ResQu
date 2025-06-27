<h2 align="center"> <a href="https://arxiv.org/abs/2505.00334">[IJCNN 2025] ResQu: Quaternion Wavelet-Conditioned Diffusion Models for Image Super-Resolution</a></h2>

<div align=center><img src="assets/network.png" width="500px"/></div>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update. </h2>


<h5 align="center">
     
 
[![arXiv](https://img.shields.io/badge/Arxiv-2505.00334-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2505.00334)
[![IEEE Explore](https://img.shields.io/badge/IEEE-Explore-blue)](https://ieeexplore.ieee.org/document/YOUR_IJCNN_DOI_HERE)

[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/Fascetta/ResQu/blob/main/LICENSE)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=Fascetta/ResQu)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/Fascetta/ResQu)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-closed/Fascetta/ResQu)

<br>
</h5>

[Luigi Sigillo](https://luigisigillo.github.io/), [Christian Bianchi](), [Aurelio Uncini](https://www.uncini.com/), and [Danilo Comminiello](https://danilocomminiello.site.uniroma1.it/home)

[ISPAMM Lab](https://ispamm.it/), Sapienza University of Rome

## 📰 News
* **[2025.07.05]** Presented the work at IJCNN 2025 in Rome!
* **[2025.06.05]** Checkpoints and code are released!
* **[2025.05.05]** The paper has been published on Arxiv 🎉. The pdf version is available [here](https://arxiv.org/abs/2505.00334)!
* **[2025.03.31]** The paper has been accepted for presentation at IJCNN 2025 🎉!

## 😮 Highlights

### 💡 Elevating Image Super-Resolution with Quaternion Wavelets
Our work introduces ResQu, a novel approach that significantly advances image super-resolution by leveraging the power of quaternion wavelet embeddings. This allows for superior feature representation, leading to high-fidelity reconstructions and enhanced perceptual quality, a crucial step in various computer vision applications.

### 🔥 State-of-the-Art Performance with Novel Conditioning
We propose a streamlined framework that conditions a latent diffusion model (built upon the StableSR baseline) using quaternion wavelet embeddings. Through extensive experimentation, ResQu demonstrates a **+15% PSNR improvement** over traditional super-resolution models, showcasing its state-of-the-art capabilities in capturing intricate texture details.

### 👀 A Multi-Scale and Frequency-Aware Approach
Unlike existing methods that demand heavy preprocessing, complex architectures, and additional components like captioning models, our approach is efficient and straightforward. This enables a new frontier in real-time BCIs, advancing tasks like visual cue decoding and future neuroimaging applications.

## 🚀 Main Results

<div align=center><img src=assets/resqu_comparison.png width="75%" height="75%"></div>
<div align=center><img src=assets/results_table_resqu.png" width="75%" height="75%"></div>

For more evaluation, please refer to our [paper](https://arxiv.org/abs/2505.00334) for details.

## How to run experiments :computer:

### Building Environment
```bash
conda create --name=resqu python=3.9
conda activate resqu
````

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install diffusers transformers accelerate xformers==0.0.16 wandb numpy==1.26.4 datasets scikit-learn torchmetrics==1.4.1 scikit-image pytorch_fid
```

### Train

To launch the training of the model, you can use the following command, you need to change the output\_dir and also specify the gpu number you want to use, right now only 1 GPU is supported:

```bash
CUDA_VISIBLE_DEVICES=N accelerate launch src/resqu/train_resqu.py \
    --pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1-base \
    --output_dir=output/resqu_model_out \
    --dataset_name=your_huggingface_dataset_name \
    --image_column=image \
    --conditioning_column=quaternion_wavelet_embedding \
    --resolution=512 \
    --learning_rate=1e-5 \
    --train_batch_size=8 \
    --num_train_epochs=50 \
    --tracker_project_name=resqu \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=1000 \
    --validation_steps=500 \
    --report_to wandb
```

### Generate

Request access to the pretrained models from [Google Drive](https://www.google.com/search?q=https://forms.gle/YOUR_GOOGLE_DRIVE_LINK_HERE).

To launch the generation of the images from the model, you can use the following commands:

```bash
CUDA_VISIBLE_DEVICES=N python src/resqu/generate_resqu.py \
    --model_path=output/resqu_model_out/checkpoint-XXXXX/ \
    --input_low_res_image_path=path/to/your/low_res_image.png \
    --output_dir=generated_images/
```

### Evaluation

Request access to the pretrained models from [Google Drive](https://www.google.com/search?q=https://forms.gle/YOUR_GOOGLE_DRIVE_LINK_HERE).

To launch the testing of the model, you can use the following command, you need to change the output\_dir:

```bash
CUDA_VISIBLE_DEVICES=N python src/resqu/evaluation/evaluate.py \
    --generated_images_path=generated_images/ \
    --ground_truth_images_path=path/to/your/ground_truth_images/
```

## Cite

Please cite our work if you found it useful:

```bibtex
@misc{sigillo2025quaternionwaveletconditioneddiffusionmodels,
      title={Quaternion Wavelet-Conditioned Diffusion Models for Image Super-Resolution}, 
      author={Luigi Sigillo and Christian Bianchi and Aurelio Uncini and Danilo Comminiello},
      year={2025},
      eprint={2505.00334},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.00334}, 
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Fascetta/ResQu&type=Date)](https://www.star-history.com/#Fascetta/ResQu&Date)

## Acknowledgement

This project is based on [StableSR](https://www.google.com/search?q=https://github.com/luojinglin/StableSR) baseline. Thanks for their awesome work.
