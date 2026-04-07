# VISTA
<p align="center">
    <a href="https://github.com/gitUmaru/syde-675-final-project" target="_blank">
    <img align="center" alt="Hymap-logo" src="https://i.imgur.com/6X3sQFb.png" width="150" height="auto"/>
    </a>
</p>
<p align="center">
<b>Vi</b>sualizing <b>S</b>pectral <b>T</b>ransformer <b>A</b>ttention: Towards Explainable Wavelength Dependencies in Hyperspectral Unmixing Transformers. This project focuses on hyperspectral unmixing, which is the process of decomposing hyperspectral imaging (HSI) response measurements into their constituent material spectral signatures (termed endmembers) and their corresponding fractional abundances.
</p>

---

## 📖 Project Overview

While deep learning models, particularly vision transformers, have shown significant promise due to their ability to capture long-range sequential dependencies across spectral bands , they suffer from an inherent "black-box" nature. 

The inability to interpret the mechanistic logic behind predictions is especially problematic when dealing with the high dimensionality of HSI data. Redundant or noisy bands drastically increase the computational burden and can actively degrade model performance. 

To address this, we aim to train a vision transformer model to analyze high-dimensional HSI data on the HyMars benchmark dataset. We then apply Explainable AI (XAI) frameworks capable of identifying and filtering spectral noise while providing physically meaningful explanations for the model's predictions.

## 🎯 Research Objectives

This project is driven by three primary objectives:
1. <b>Model Development<b>: Develop a vision transformer to accurately decompose HSI pixels into their constituent endmembers and fractional abundances.
2. <b>Interpretability<b>: Apply explainable AI to uncover the model's prediction logic and identify the most critical spectral bands.
3. <b>Dimensionality Reduction<b>: Utilize these insights to filter out noisy bands, lowering computational complexity without sacrificing performance.

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
