# 🖥️ Machine Learning Model Training App

This app provides a user-friendly interface to upload datasets, select machine learning models, and visualize classification results such as confusion matrix, ROC curve, and accuracy. It includes explanations for each of the key performance metrics.

## 🚀 Features

- **Upload Dataset**: Upload your own CSV file for analysis, or use the provided breast cancer dataset.
- **Feature Selection**: Choose which columns in your dataset should be used as features and which as the label.
- **Model Selection**: Choose from four machine learning models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Multilayer Perceptron (MLP)
- **Classification Metrics**: Automatically generate and view the following:
  - **Accuracy Score**
  - **Classification Report** (Precision, Recall, F1-Score)
  - **Confusion Matrix** (displayed as a heatmap)
  - **ROC Curve** (with AUC)

## 🛠️ Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ml-streamlit-app.git
cd ml-streamlit-app
