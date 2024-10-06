import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.datasets import load_breast_cancer

# Define the model dictionary
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Support Vector Machine": SVC(random_state=42, probability=True),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multilayer Perceptron": MLPClassifier(random_state=42, max_iter=300)
}

# Load sklearn cancer classification dataset
def load_demo_data():
    cancer = load_breast_cancer()
    data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    data['label'] = cancer.target
    return data

# Streamlit APP title
st.title("🖥️ Machine Learning Model Training App")

# Display description of the app
st.write("""
This app allows you to train and evaluate different machine learning models on your own dataset or use the DEMO dataset.🤩
""")

# Upload or select DEMO dataset
st.write("Upload your dataset or choose the DEMO dataset to proceed.")

# Create session state to track whether the user has uploaded data or chosen the demo file
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

# Initialize `data`
data = None

# Show the upload button or use demo button only if no data is loaded
if not st.session_state['data_loaded']:
    # Upload data
    uploaded_file = st.file_uploader("⬇️Upload your CSV file", type=["csv"])

    # Use the DEMO file if selected
    if st.button("Use DEMO File"):
        data = load_demo_data()
        st.write("Using the Breast Cancer DEMO dataset:")
        st.write(data)
        st.session_state['data_loaded'] = True  
        st.session_state['data'] = data
    elif uploaded_file is not None:
        try:
            # Read user-uploaded data
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded dataset:")
            st.write(data)
            st.session_state['data_loaded'] = True  
            st.session_state['data'] = data
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
else:
    st.write("### Data successfully loaded!🎉🎉🎉")
    data = st.session_state['data']

# Feature selection and model training (only show this if data is loaded)
if st.session_state['data_loaded'] and data is not None:
    st.write("### 🏷️Select Features and Label")
    
    
    features = st.multiselect("⬇️Select Feature Columns", options=data.columns.tolist())
    label = st.selectbox("⬇️Select Label Column", options=data.columns.tolist())
    
    if features and label:
        if label in features:
            st.error("The label cannot be one of the selected features. Please adjust your selection.")
        else:
            X = data[features]
            y = data[label]

            if y.dtype in [np.float64, np.int64] and len(y.unique()) > 2:
                st.warning("It seems your label is continuous. Please convert it to categorical before training.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Select a machine learning model
                st.write("### 🔷Select a Machine Learning Model")
                model_choice = st.selectbox("⬇️Choose a model", list(models.keys()))

                # Train the model
                if st.button("Train Model"):
                    model = models[model_choice]

                    try:
                        # Train the selected model
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_prob = model.predict_proba(X_test)[:, 1]  

                        # Display results
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"### 📌Accuracy: {accuracy:.2f}")

                        st.write("""
                         **💠What does this accuracy mean?**
                        
                        The accuracy is the proportion of correctly predicted instances out of all instances. A higher accuracy indicates that the model is better at predicting the correct class. However, accuracy alone can be misleading if the classes are imbalanced, which is why we also look at other metrics like precision, recall, and the confusion matrix.
                        """)

                    

                        # Display the results of classification
                        st.write("### 📝Classification Report")
                        st.write("The classification report provides detailed metrics for each class, including precision, recall, and F1-score:")
                        st.text(classification_report(y_test, y_pred))

                        # Explain the classification Report
                        st.write("""
                        **💠Understanding the Classification Report:**
                        
                        - **Precision**: The ratio of correctly predicted positive observations to the total predicted positives. High precision indicates a low false positive rate.
                        - **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class. High recall indicates a low false negative rate.
                        - **F1-Score**: The weighted average of precision and recall. It is useful when you want a balance between precision and recall.
                        """)

                    

                        # Confusion Matrix
                        cm = confusion_matrix(y_test, y_pred)
                        st.write("### 📊Confusion Matrix:")
                        st.write("The confusion matrix shows the number of correct and incorrect predictions made by the model:")
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        st.pyplot(fig)

                         # Explain confusion matrix
                        st.write("""
                        **💠Understanding the Confusion Matrix:**
                        
                        - **True Positives (TP)**: The model correctly predicted the positive class.
                        - **True Negatives (TN)**: The model correctly predicted the negative class.
                        - **False Positives (FP)**: The model incorrectly predicted the positive class (Type I error).
                        - **False Negatives (FN)**: The model incorrectly predicted the negative class (Type II error).
                        
                        A good model has high TP and TN values, and low FP and FN values.
                        """)

                        # ROC Curve
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        st.write("### 📈ROC Curve")
                        st.write("The ROC curve shows the trade-off between sensitivity (recall) and specificity for every possible classification threshold:")
                        st.write(f"**ROC AUC**: {roc_auc:.2f}")
                        fig_roc, ax_roc = plt.subplots()
                        ax_roc.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                        ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
                        ax_roc.set_xlim([0.0, 1.0])
                        ax_roc.set_ylim([0.0, 1.05])
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                        ax_roc.legend(loc="lower right")
                        st.pyplot(fig_roc)

                        # Explain ROC curve
                        st.write("""
                        **💠Interpreting the ROC Curve:**
                        
                        The ROC curve plots the true positive rate (recall) against the false positive rate (1 - specificity) at various threshold settings.
                        - **AUC (Area Under the Curve)**: AUC provides an aggregate measure of the model's performance across all classification thresholds. A higher AUC value indicates a better performing model.
                        """)

                    except ValueError as e:
                        st.error(f"Error during model training: {e}")
                    