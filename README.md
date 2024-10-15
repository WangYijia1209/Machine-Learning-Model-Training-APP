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
```

### 2. Install Dependencies

Ensure that you have Python 3.7+ installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
To launch the app, run the following command:

```bash
streamlit run app.py
```

This will open the app in your browser at http://localhost:8501.

## 📝How It Works
-**1.Upload or Use Demo Data**: Upload your CSV file or use the built-in breast cancer dataset.
-**2.Select Features and Label**: Choose which columns in your dataset are features and which column is the label (target variable).
-**3.Model Training**: Select a machine learning model to train.
-**4.View Results**: Once the model is trained, the app will display:
  -**-Accuracy**: Proportion of correctly predicted instances.
  -**-Classification Report**: Provides detailed metrics for each class.
  -**-Confusion Matrix**: Shows the true positives, true negatives, false positives, and false negatives.
  -**-ROC Curve**: A graphical representation of the model's performance, with the AUC value provided.

## 🎉 Example Output
-**Confusion Matrix:** Visualized as a heatmap to show correct and incorrect predictions.
-**ROC Curve**: Displays the trade-off between sensitivity and specificity, with the Area Under the Curve (AUC) provided.

## 💡 Interpreting the Results
-**Accuracy**: A higher accuracy indicates better model performance but should be considered along with other metrics, especially for imbalanced data.
-**Classification Report**: Explains precision, recall, and F1-score for each class.
-**Confusion Matrix**: Helps understand where the model makes errors.
-**ROC Curve**: Provides insights into the performance across different thresholds.



