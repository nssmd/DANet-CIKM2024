# DANet: A RAG-inspired Dual Attention Model for Few-shot Time Series Prediction

## Introduction

DANet is a novel few-shot time series forecasting model inspired by the **Retrieval-Augmented Generation (RAG)** framework. It is designed to enhance prediction accuracy by retrieving and integrating **long-term time series (LTS)** and **short-term time series (STS)** based on pattern similarities.

## Key Features

- **RAG-inspired architecture**: Utilizes retrieval-based techniques to identify relevant patterns in LTS for STS forecasting.
- **Dual Attention Mechanism**: Embeds both **global similarity** (frequency-based) and **local similarity** (fluctuation-based) to improve feature extraction.
- **Few-shot Learning Capability**: Effectively forecasts STS with limited data by leveraging multiple related LTS.
- **Efficient Computation**: Optimized with attention-based mechanisms to handle large-scale time series data efficiently.

## Model Architecture

DANet consists of three main components:

1. **Retrieval Module**: Identifies and retrieves LTS sequences similar to the given STS.
2. **Augmentation Module**: Uses a **softmax-weighted** mechanism to refine retrieved patterns.
3. **Generation Module**: Employs feed-forward layers to generate accurate time series predictions.

<p align="center">
  <a href="https://buymeacoffee.com/fernandezowen" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-red.png" alt="Buy Me A Coffee" width="150">
  </a>
</p>

## Installation

To use DANet, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To train the model, use:

```bash
python train.py --dataset <dataset_name> --epochs 300 --batch_size 128
```

For evaluation:

```bash
python evaluate.py --dataset <dataset_name>
```

## Datasets

DANet has been evaluated on six real-world datasets:

- **Solar_h, Solar_m** (Solar energy forecasting)
- **Traffic** (Transportation data)
- **Electricity** (Power consumption forecasting)
- **Parking** (Parking lot availability)
- **Weather** (Meteorological data)

## Benchmark Results

DANet significantly outperforms six state-of-the-art models:

| Dataset | Model | MSE ↓ | MAE ↓ |
|---------|-------|------|------|
| Parking | DANet | **0.024** | **0.077** |
| Traffic | DANet | **0.184** | **0.167** |
| Weather | DANet | **0.405** | **0.493** |

For full benchmark results, please refer to our paper.

## Citation

If you find this work useful, please cite:

```
@article{DANet2025,
  author    = {Your Name and Others},
  title     = {DANet: A RAG-inspired Dual Attention Model for Few-shot Time Series Prediction},
  journal   = {Under review at UAI 2025},
  year      = {2025}
}
```

## Support

If you like this project, consider **supporting us** by buying a coffee! ☕  
Click the button below:

<p align="center">
  <a href="https://buymeacoffee.com/fernandezowen" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-red.png" alt="Buy Me A Coffee" width="150">
  </a>
</p>
