# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, stats
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from numpy import mean
import pickle
import warnings
import json
import os
warnings.filterwarnings('ignore')

class HeartDiseaseML:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.results_file = "dashboard_data.json"  # JSON file to log results

    def log_to_json(self, step, details):
        """Log step results to a JSON file."""
        log_data = {step: details}

        # Check if file exists, then append or create
        if os.path.exists(self.results_file):
            with open(self.results_file, "r") as file:
                existing_data = json.load(file)
        else:
            existing_data = {}

        # Update existing data with new logs
        existing_data.update(log_data)

        with open(self.results_file, "w") as file:
            json.dump(existing_data, file, indent=4)

    def load_and_clean_data(self):
        self.data = pd.read_csv(self.data_path)
        self.data = self.data.rename(
            columns={'cp': 'chest_pain_type',
                     'trestbps': 'resting_blood_pressure',
                     'chol': 'cholesterol',
                     'fbs': 'fasting_blood_sugar',
                     'restecg': 'resting_electrocardiogram',
                     'thalach': 'max_heart_rate_achieved',
                     'exang': 'exercise_induced_angina',
                     'oldpeak': 'st_depression',
                     'slope': 'st_slope',
                     'ca': 'num_major_vessels',
                     'thal': 'thalassemia'}
        )
        print("Data loaded and cleaned successfully!")
        self.log_to_json("load_and_clean_data", {"status": "completed"})

    def explore_data(self):
        print("Data Info:\n", self.data.info())
        print("Data Description:\n", self.data.describe())
        print("Null Values:\n", self.data.isnull().sum())
        print("Correlation Heatmap:")
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.show()
        self.log_to_json("explore_data", {"status": "completed"})

    def preprocess_data(self):
        self.data.drop(['resting_blood_pressure', 'cholesterol',
                        'fasting_blood_sugar', 'resting_electrocardiogram'], axis=1, inplace=True)
        self.X = self.data[['age', 'sex']]
        self.Y = self.data[['target']]
        self.data = self.data.drop(['age', 'sex', 'target'], axis=1)

        pt = PowerTransformer(method='yeo-johnson')
        for col in ['chest_pain_type', 'max_heart_rate_achieved', 'st_depression',
                    'num_major_vessels', 'thalassemia']:
            self.data[col] = pt.fit_transform(self.data[[col]])

        scaler = MinMaxScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        self.data = pd.concat([self.X, self.data], axis=1)
        print("Data Preprocessing Completed!")
        self.log_to_json("preprocess_data", {"status": "completed"})

    def split_data(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.data, self.Y, test_size=0.2, random_state=123456)
        print("Data split into training and testing sets!")
        self.log_to_json("split_data", {
            "X_train_shape": self.X_train.shape,
            "X_test_shape": self.X_test.shape,
            "Y_train_shape": self.Y_train.shape,
            "Y_test_shape": self.Y_test.shape
        })

    def train_models(self):
        logistic_model = LogisticRegression()
        logistic_model.fit(self.X_train, self.Y_train)

        knn_model = KNeighborsClassifier(n_neighbors=3)
        knn_model.fit(self.X_train, self.Y_train)

        svm_model = SVC(probability=True)
        svm_model.fit(self.X_train, self.Y_train)

        print("Models trained successfully!")
        self.log_to_json("train_models", {"status": "completed"})
        return logistic_model, knn_model, svm_model

    def evaluate_model(self, model, model_name):
        y_pred = model.predict(self.X_test)
        y_pred_binary = [1 if i > 0.5 else 0 for i in y_pred]
        cm = confusion_matrix(self.Y_test, y_pred_binary)
        acc = accuracy_score(self.Y_test, y_pred_binary)
        recall = recall_score(self.Y_test, y_pred_binary)
        precision = precision_score(self.Y_test, y_pred_binary)

        print(f"<-------------------{model_name}------------------->")
        print('Confusion matrix:\n', cm)
        print(f"Accuracy: {acc}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")

        # Log evaluation metrics
        self.log_to_json(f"evaluate_model_{model_name}", {
            "confusion_matrix": cm.tolist(),
            "accuracy": acc,
            "recall": recall,
            "precision": precision
        })

    def save_model(self, model, filename):
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved as {filename}!")
        self.log_to_json("save_model", {"filename": filename, "status": "completed"})

# Usage
data_path = r"C:\Users\anike\PycharmProjects\MachineLearning_CA1\test\heart-disease-dataset\heart.csv"

