
# Diabetes Prediction Using Perceptron  
*A Deep Learning Fundamentals Assignment*

## Overview
This project implements a **single-layer Perceptron** to classify individuals as diabetic or non-diabetic using the [Pima Indian Diabetes dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database). The Perceptron model is trained and evaluated across various learning rates and data splits, both with and without output scaling.

## Project Structure

```
Perceptron_diabeties.ipynb   # Main notebook containing implementation, training, and evaluation
dataset/
└── diabetes.csv              # Input dataset
```

## How It Works

### 1. **Data Preprocessing**
- Standardises features using training-set statistics.
- Optionally scales output labels from `{0, 1}` to `{-1, 1}` for better gradient dynamics.

### 2. **Perceptron Architecture**
A classic McCulloch-Pitts single-neuron binary classifier:
```
Output = sign(w · x + b)
```
- Uses a threshold activation function.
- Weights and bias are updated using perceptron learning rule:
  ```
  w = w + α * x * (y_true - y_pred)
  b = b + α * (y_true - y_pred)
  ```

### 3. **Training & Evaluation**
- Performs multiple training sessions over configurable `learning_rate`, `epochs`, and `test_size`.
- Logs training/testing accuracy per epoch.
- Option to experiment with output scaling for performance tuning.

## 🔬 Experimental Findings

| Output Scaling | Learning Rate | Accuracy (Train/Test) |
|----------------|----------------|------------------------|
|  No           | 0.001 - 0.5     | ~55–79%                |
|  Yes          | 0.001 - 0.1     | **~100%** (at 0.01–0.1) |

Output scaling dramatically improved classification performance, reaching **100% accuracy** under the right hyperparameters.

## 🛠️ Requirements
- Python 3.x
- Jupyter Notebook
- Libraries: `numpy`, `pandas`, `matplotlib`

```bash
pip install numpy pandas matplotlib
```

## 📊 Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/ChinmayDharmik/DLF_assign_1_perceptron.git
   cd DLF_assign_1_perceptron
   ```

2. Place `diabetes.csv` into a `dataset/` folder if not already present.

3. Run the notebook:
   ```bash
   jupyter notebook Perceptron_diabeties.ipynb
   ```

## Results Summary

- Output scaling improves convergence drastically.
- Best performance: **100% accuracy at `lr=0.1`, `epochs=10`** with scaled labels.
- Demonstrates the potential of even simple models when correctly tuned.
