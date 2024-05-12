# Image Enhancement with SRGAN and Autoencoder

This repository contains the implementation of two distinct image processing models: Super-Resolution Generative Adversarial Network (SRGAN) and an Autoencoder, focused on enhancing blurred and pixelated face images generated by a GAN model. The project demonstrates the effectiveness of SRGAN over Autoencoder in achieving superior image clarity.

## Project Overview

The goal of this project is to improve the quality of blurred and pixelated face images. It compares two methodologies:
1. **SRGAN**: Utilizes a generative adversarial network to enhance low-resolution images into high-resolution images. It is particularly effective due to its adversarial training approach, where the generator and discriminator compete, leading to higher quality outputs.
2. **Autoencoder**: Aims to learn a compressed representation of images and then reconstruct them to a higher quality. Although beneficial for many tasks, it may not handle the fine details and textures as effectively as SRGAN in super-resolution contexts.

## Comparison
![boy](https://github.com/nick860/SRGAN_Vs_Auto_CelebA/assets/55057278/0ce07cf2-095c-4a04-b310-4c8b732d11b3)
![girl](https://github.com/nick860/SRGAN_Vs_Auto_CelebA/assets/55057278/c8a76f6e-313d-427d-87ec-0c746152c00b)

The project includes a detailed comparison of these models, highlighting how SRGAN outperforms the autoencoder in terms of:
- Detail preservation
- Texture enhancement
- Overall image clarity
