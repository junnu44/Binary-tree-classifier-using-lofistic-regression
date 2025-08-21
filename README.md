📌 Project Overview
This project uses Logistic Regression to classify tumors as Malignant (cancerous) or Benign (non-cancerous) based on medical diagnostic measurements.

The dataset comes from the Breast Cancer Wisconsin Dataset, which is a popular benchmark dataset for binary classification problems. Logistic Regression is chosen for its simplicity, interpretability, and effectiveness in binary classification tasks.

🎯 Objectives
Build a binary classification model using Logistic Regression.

Train the model on the Breast Cancer dataset.

Evaluate the model using accuracy, precision, recall, F1-score, and ROC-AUC.

Visualize results using a confusion matrix and ROC curve.

🛠️ Tech Stack
Programming Language: Python 🐍

Libraries:

NumPy → numerical computations

Pandas → data manipulation

Matplotlib & Seaborn → visualization

Scikit-learn → dataset, preprocessing, model building, evaluation

📂 Dataset
The Breast Cancer Wisconsin Dataset contains:

Features (30): Measurements such as mean radius, mean texture, mean smoothness, etc.

Target:

0 → Malignant (cancerous)

1 → Benign (non-cancerous)

Dataset is directly available in scikit-learn:

python
Copy
Edit
from sklearn.datasets import load_breast_cancer
🔎 Methodology
Load Dataset

Import dataset using load_breast_cancer() from sklearn.

Data Preprocessing

Standardize features using StandardScaler.

Split dataset into training (80%) and testing (20%) sets.

Model Training

Train a Logistic Regression model on the training set.

Model Evaluation

Accuracy Score

Precision, Recall, F1-score

Confusion Matrix

ROC Curve & AUC

Prediction

Predict tumor type (Malignant/Benign) on unseen test data.

📊 Results
Logistic Regression achieves high accuracy (typically ~95-97%).

Confusion Matrix shows classification performance.

ROC Curve demonstrates strong separation between classes.

Example Evaluation Metrics:

makefile
Copy
Edit
Accuracy: 0.96
Precision: 0.97
Recall: 0.96
F1-score: 0.96
🚀 How to Run the Project
Clone this repository:

bash
Copy
Edit
git clone https://github.com/junnu44/Binary-tree-classifier-using-logistic-regression.git
cd breast-cancer-logistic-regression
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter Notebook:

bas
Copy
Edit
jupyter notebook Breast_Cancer_Logistic_Regression.ipynb
📌 Future Improvements
Apply feature selection to improve model interpretability.

Try other classifiers: Random Forest, SVM, Gradient Boosting.

Build a Streamlit app to allow doctors/patients to input tumor measurements and predict outcomes.

✨ Conclusion
This project shows how Logistic Regression can be effectively used for Breast Cancer classification. The model achieves strong performance, demonstrating how machine learning can support healthcare by aiding in early detection of cancer.
