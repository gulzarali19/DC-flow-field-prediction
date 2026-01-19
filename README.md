# DC-Flow-Field-Prediction

**Real-Time Thermal Field Prediction in Data Centers Using Deep Learning Surrogates for CFD**

This repository contains a framework for predicting high-resolution thermal fields in data centers using deep learning-based surrogate models. By utilizing 2D Convolutional Neural Networks (CNNs), this approach achieves prediction speeds over **6,000 times faster** than traditional Computational Fluid Dynamics (CFD) simulations—reducing computation time from ~4 minutes to just **0.07 seconds**—while maintaining high spatial accuracy.

## 📌 Project Overview

Monitoring thermal environments in data centers (DC) is critical for identifying hotspots that could lead to equipment failure. While CFD provides accurate thermal maps, its high computational cost makes it unsuitable for real-time monitoring.

This project introduces a lightweight decoder-style CNN that predicts full resolution temperature fields using only three low-dimensional scalar inputs:

1. **Server Workload** (Power: 416W to 1666W)
2. **ACU Airflow Rate** (Velocity: 1.75 to 2.65 m/s)
3. **Temperature Setpoints** (290K to 299K)

## 🚀 Key Features

* **Real-Time Performance:** Inference completed in **0.07 seconds** per sample.
* **High Fidelity:** Uses a **hybrid loss function** combining  (MAE) for pixel accuracy and **Perceptual Loss (VGG-19)** to preserve structural and textural details.
* **Efficient Architecture:** The proposed **Normalized Deconv Decoder** utilizes Batch Normalization to stabilize training and improve feature learning.

## 🛠 Methodology

### Data Generation

The ground truth dataset was generated using **ANSYS Fluent** with a steady-state, pressure-based coupled solver and the standard  turbulence model.

* **Total Simulations:** 2,000 unique parametric cases.
* **Geometry:** A 2D representation of a data center rack housing 12 servers.
* **Grid Independence:** Verified convergence at a mesh density of 43,836 nodes.

### Neural Network Architectures

Three architectures were evaluated:

* **Model 1 (Simple Deconv Decoder):** Baseline mapping scalars to images through a streamlined decoding pipeline.
* **Model 2 (Residual Deconv Decoder):** Deeper architecture using **residual blocks** and identity skip connections.
* **Model 3 (Normalized Deconv Decoder):** The proposed optimized model using **Batch Normalization** and transposed convolutional layers.

## 📊 Performance Results

Evaluated on a 10% held-out test set:

| Metric | Model 1 | Model 2 | **Model 3 (Best)** |
| --- | --- | --- | --- |
| **Avg. L1 Loss (MAE)** | 0.0666 | 0.0754 | **0.0391** |
| **Avg. PSNR (dB)** | 19.70 | 18.09 | **24.45** |
| **Avg. SSIM** | 0.7767 | 0.8079 | **0.8575** |
| **Avg. Perceptual Loss** | 0.7536 | 0.7626 | **0.4659** |

## 📂 Repository Structure

* `src/models/`: PyTorch implementations of the Deconv, Residual, and Normalized architectures.
* `data/`: Phase-I and II Simulations data Preprocessed with 128 by 256 image size.
* `csv/` : All parameters mapped to original thermal contours
* `results/` Evaluation metrics (MSE, PSNR, SSIM) and visualization of error maps.
* `src/utils`: Python/Journal scripts used to automate contour extraction from ANSYS Fluent.
