# Differential Privacy Experiments with Machine Learning

This repository is developed to perform **experiments and benchmarks** for understanding the impact of **differential privacy** (DP) on machine learning algorithms. It leverages IBM's [Diffprivlib](https://github.com/IBM/differential-privacy-library), providing state-of-the-art methods for integrating differential privacy into machine learning workflows. Additionally, we designed custom **benchmarking pipelines** to evaluate the performance of these methods across multiple metrics.

---

## 🗂️ Project Structure

The repository is organized as follows:

DIFFERENTIAL_PRIVACY 

```
  ├── modules                    
  │   ├── dp_benchmark.py # Main benchmarking module 
  │   ├── utils.py # Utility functions for the benchmarks 
  ├── notebooks
  │   ├── dp_benchmark.ipynb # Notebook to run benchmarks and visualize results  
  │   ├── dp_exploration_and_modeling.ipynb # Exploratory analysis and DP modeling 
  └── plots
  └── .gitignore # Git ignore file
  └── LICENSE # License information 
  └── README.md # Documentation (you are here!) 
  └── requirements.txt # Dependencies for the project
```
---

## 📚 Overview

Differential Privacy (DP) is a mathematical framework that ensures privacy protection for individuals within a dataset. In this repository:

1. **We evaluate the effects of differential privacy on machine learning algorithms**, focusing on balancing privacy guarantees with model performance.
2. **We utilize IBM's Diffprivlib library** to incorporate DP mechanisms into algorithms like logistic regression and decision trees.
3. **We create custom benchmarks** to evaluate metrics such as accuracy, precision, recall, F1 score.

### Core Objectives:
- **Understand the trade-offs** between privacy (epsilon) and model performance.
- **Benchmark differential privacy algorithms** across various datasets and metrics.
- **Visualize and interpret results** to better understand the impacts of differential privacy mechanisms.

---

## 📊 Benchmarks and Metrics

Benchmarks are designed to compare **non-private** machine learning models with **differentially private** ones under varying privacy budgets (epsilon). Results are stored and visualized in the `plots/` directory.

---

## 📂 How to Use

### Prerequisites
Make sure you have **Python 3.9** installed. You can create a Conda environment for this project:
```bash
conda create --name dp_env python=3.9
conda activate dp_env
