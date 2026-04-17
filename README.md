# VISTAE
<p align="center">
    <a href="https://github.com/gitUmaru/syde-675-final-project" target="_blank">
    <img align="center" alt="Hymap-logo" src="https://i.imgur.com/6X3sQFb.png" width="150" height="auto"/>
    </a>
</p>
<p align="center">
<b>Vi</b>sion <b>S</b>pectral <b>T</b>ransformer for <b>A</b>bundance <b>E</b>stimation
</p>

---

## 📖 Project Overview

Hyperspectral imaging (HSI) provides highly detailed electromagnetic responses across hundreds of closely sampled wavelengths, enabling precise discrimination of surface mineralogy. A fundamental objective in HSI analysis is hyperspectral unmixing: the process of decomposing these response measurements into their constituent material spectral signatures (endmembers) and their corresponding fractional abundances. 

A major practical challenge in deep learning-based hyperspectral unmixing is the limited availability of reliable, continuous ground-truth abundance maps. While discrete classification datasets are more readily accessible, they fail to capture the inherently mixed nature of hyperspectral pixels. 

**VISTAE** bridges the gap between discrete image classification and continuous abundance estimation. By reformulating traditional multi-class classification as a soft-clustering problem, this project leverages sparse classification datasets as weak priors. We implement a vision transformer-based autoencoder architecture to infer latent endmembers and extract continuous, sub-pixel fractional abundances, remaining robust to incomplete labeling and unknown material compositions. 

## 🎯 Research Objectives

This repository contains the codebase and experimental framework designed to achieve the following core objectives:

1. **Latent Endmember Learning:** Develop a transformer-based model that performs robust hyperspectral unmixing through unsupervised spectral reconstruction and soft abundance estimation.
2. **Weakly Supervised Prior Integration:** Incorporate sparse, discrete classification labels as weak priors to actively guide the soft-clustering process, anchoring the unconstrained latent space to known geological realities.
3. **Comparative Benchmarking:** Evaluate the efficacy of the proposed spatial-spectral architecture against established classical baselines (e.g., Vertex Component Analysis) and state-of-the-art hybrid models (e.g., DeepTrans-HSU) on complex, heavily mixed planetary datasets.

## 👥 Authors
* **Anthony Bertnyk** - Department of Systems Design Engineering, University of Waterloo
* **Muhammad Umar Ali** - Department of Systems Design Engineering, University of Waterloo 

## 🗂️ Repository Structure

```text
.
├── data/
│   ├── data_description.xlsx
│   ├── holden_gt.mat
│   ├── holden.mat
│   ├── Niilifossae_gt.mat
│   ├── Niilifossae.mat
│   ├── Utopia_gt.mat
│   └── Utopia.mat
├── experiments/
│   ├── __init__.py
│   └── base-experiment.py
├── models/
│   └── __init__.py
├── utils/
│   ├── __init__.py
│   └── logger.py
├── venv/
├── .gitignore
├── README.md
└── requirements.txt
```

## 📊 Dataset: HyMars Benchmark

The project utilizes the HyMars hyperspectral image classification benchmark dataset. The hyperspectral data for these three Martian scenes was captured by the **Compact Reconnaissance Imaging Spectrometer for Mars (CRISM)** instrument aboard the **Mars Reconnaissance Orbiter (MRO)**. 

Each region presents unique geological characteristics, and the datasets have been preprocessed to filter out water absorption and noisy spectra. 

* **Holden Crater (HC)**
    * 418 × 595 pixels, 440 spectral bands
    * Notable for its fluviolacustrine (river and lake) geological evolution. Analyzing this region helps investigate historical Martian water level changes and timing.
* **Nili Fossae (NF)**
    * 478 × 593 pixels, 425 spectral bands
    * Unique for containing specific mineral phases like carbonates and serpentine. This suggests possible historical hydrothermal activity, which is an important factor when evaluating the potential for past biological activity on Mars.
* **Utopia Planitia (UP)**
    * 478 × 595 pixels, 432 spectral bands
    * Geographically significant as the landing site for the Tianwen-1 rover, 'Zhurong'.
    
### File Structure
All dataset files are housed within the `/data` directory:
* **`.mat` files:** Contain the raw hyperspectral response measurements across the spectral bands.
* **`_gt.mat` files:** Contain the annotated ground-truth labels for evaluating the model. 
  * *Important Note on Labels:* This dataset is sparsely labeled. Instead of every pixel containing a fractional abundance, only a specific subset of pixels contains a numerical value (1-9) indicating a single, dominant mineral class. Unlabeled pixels act as background. For the purposes of unmixing, these discretely labeled pixels serve as pure endmember signatures.
* **`data_description.xlsx`:** Provides supplementary metadata and structural information about the samples.

> **Reference:** > B. Xi, Y. Zhang, J. Li, T. Zheng, X. Zhao, H. Xu, C. Xue, Y. Li, and J. Chanussot, "MCTGCL: Mixed CNN–Transformer for Mars Hyperspectral Image Classification With Graph Contrastive Learning" *IEEE Transactions on Geoscience and Remote Sensing*, vol. 63, pp. 1-14, 2025.
