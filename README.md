# Semi-Supervised Learning with PSO-Optimized Pseudo-Labeling on CIFAR-10

> Combining **Pseudo-Label SSL** and **Particle Swarm Optimization** to maximize image classification accuracy when labeled data is scarce.

---

## Overview

This project tackles a core challenge in real-world ML: **what do you do when you only have labels for a small fraction of your data?**

We implement a **weighted pseudo-labeling pipeline** that iteratively assigns confident model predictions as labels to unlabeled samples — then uses **Particle Swarm Optimization (PSO)** to automatically find the best hyperparameters for that process.

Built with **PyTorch** on the **CIFAR-10** benchmark dataset.

---

## Key Features

- **Semi-Supervised Learning (SSL)** — trains on only 10–20% labeled data, bootstraps the rest
- **Confidence-thresholded Pseudo-Labeling** — only high-confidence predictions are added as pseudo-labels, preventing noisy label propagation
- **Weighted Loss** — pseudo-labeled samples are down-weighted during training to reduce their influence relative to ground-truth labels
- **Particle Swarm Optimization (PSO)** — swarm-based metaheuristic that searches the hyperparameter space (`threshold`, `max_pseudo_labels`, `pseudo_weight`) to maximize validation accuracy
- **Macro F1 Evaluation** — robust to class imbalance, reported alongside accuracy

---

## Architecture

```
Input (CIFAR-10 32x32 RGB)
    │
    ▼
SimpleCNN
├── Conv2d(3→32) + ReLU + MaxPool
├── Conv2d(32→64) + ReLU + MaxPool
├── Flatten
├── Linear(4096→128) + ReLU
└── Linear(128→10)
    │
    ▼
PSO Hyperparameter Optimizer
├── Particle: [threshold, max_pseudo_labels, pseudo_weight]
├── Swarm size: configurable (default 5)
└── Fitness: validation accuracy
```

---

## How It Works

### 1. Initial Training
The model is trained on a small labeled subset (e.g., 10% of training data).

### 2. Pseudo-Label Generation
For each unlabeled sample, the model predicts a probability distribution. Samples where the maximum probability exceeds a **confidence threshold** (e.g., 0.97) are assigned pseudo-labels.

### 3. Weighted Retraining
Pseudo-labeled samples are added to the training set with a reduced weight (`pseudo_weight < 1.0`), so the model trusts ground-truth labels more.

### 4. PSO Optimization
PSO explores the hyperparameter space across multiple rounds, updating particle positions using cognitive and social forces to find the combination that maximizes validation accuracy.

---

## Results

| Configuration | Val Accuracy | Notes |
|---|---|---|
| Supervised (10% labels) | Baseline | No pseudo-labels |
| SSL Round 1 | +2–4% | Initial pseudo-labels added |
| SSL Round 3 (best) | Peak | Best model saved |

> PSO converges to optimal `threshold`, `max_pseudo_labels`, and `pseudo_weight` automatically — no manual tuning needed.

---

## Project Structure

```
COSC4P96-GroupProject/
├── model.py               # SimpleCNN architecture
├── data.py                # CIFAR-10 data loading & splitting
├── ssl_pipeline.py        # Weighted pseudo-label SSL pipeline
├── train_supervised.py    # Baseline supervised training
├── pso.py                 # Particle Swarm Optimizer
├── optimize_ssl.py        # PSO entry point
├── plot_pso_history.py    # PSO convergence visualization
├── config.py              # Shared configuration
└── pso_convergence.png    # Convergence plot
```

---

## Getting Started

### Prerequisites
```bash
pip install torch torchvision numpy
```

### Run Supervised Baseline
```bash
python train_supervised.py
```

### Run SSL Pipeline
```bash
python ssl_pipeline.py
```

### Run PSO Hyperparameter Search
```bash
python pso.py
```

### Plot PSO Convergence
```bash
python plot_pso_history.py
```

---

## Configuration

Key parameters in `ssl_pipeline.py`:

| Parameter | Default | Description |
|---|---|---|
| `labeled_ratio` | 0.10 | Fraction of training data that is labeled |
| `threshold` | 0.97 | Minimum confidence to accept a pseudo-label |
| `max_pseudo_labels_per_round` | 500 | Max new pseudo-labels per SSL round |
| `pseudo_weight` | 1.0 | Loss weight for pseudo-labeled samples |
| `ssl_rounds` | 3 | Number of SSL iterations |
| `epochs_per_round` | 4 | Training epochs per round |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-green)

---


