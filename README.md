# SWIFT: Mapping Sub-series with Wavelet Decomposition Improves Time Series Forecasting

[![arXiv](https://img.shields.io/badge/arXiv-2501.16178-B31B1B.svg)](https://arxiv.org/abs/2501.16178)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)  

SWIFT is a lightweight and efficient model for **Long-term Time Series Forecasting (LTSF)** that integrates **wavelet decomposition**, **cross-band information fusion**, and **sub-series mapping** techniques to improve forecasting accuracy, especially in resource-constrained environments.  

---

## 🚀 Features  

- **Wavelet-based Lossless Downsampling**: Uses **Discrete Wavelet Transform (DWT)** for effective time-frequency analysis.  
- **Cross-band Information Fusion**: Learns a **shared representation** across different frequency bands.  
- **Minimalist Model Design**: Employs a **single-layer linear** or **shallow MLP** for sub-series mapping.  
- **Superior Performance**: Achieves **state-of-the-art (SOTA)** results with **only 25%** of the parameters of a standard linear model.  
- **Optimized for Edge Devices**: Suitable for **low-computation environments** with fast inference speed.  

---

## 📜 Table of Contents  

- [Installation](#installation)
- [Run Experiment](#run)
- [Citing SWIFT](#citing-swift)  

---

## 🔧 Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/LancelotXWX/SWIFT.git
cd SWIFT
pip install -r requirements.txt
```

## 🔥 Run

To run the experiment in paper:
```bash
bash scripts/SWIFT/etth1.sh
bash scripts/SWIFT/traffic.sh
```

## 📌 Citing SWIFT  

If you find SWIFT useful in your research, please cite:  

```bibtex
@article{xie2025swift,
  title={SWIFT: Mapping Sub-series with Wavelet Decomposition Improves Time Series Forecasting},
  author={Xie, Wenxuan and Cao, Fanpu},
  journal={arXiv preprint arXiv:2501.16178},
  year={2025}
}
```
