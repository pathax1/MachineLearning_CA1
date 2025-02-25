import pytest
import time
import json
import os
from selenium import webdriver
import pandas as pd
from dashboard.Prediction import HeartAttackPredictorApp, predict_heart_attack
from pages.Kraggle_Repo import Kraggle_Repo
from pages.Module_Execution_ML import HeartDiseaseML
from utils.data_loader import load_test_data
from selenium.webdriver.chrome.service import Service
import logging
import subprocess
import webbrowser
import tkinter as tk

@pytest.fixture(scope="session")
def config():
    # Configuration details for the base URL
    return {
        "base_url": "https://www.kaggle.com/"  # Replace with your app's URL
    }

@pytest.fixture
def driver(config):
    service = Service("C:/Users/anike/PycharmProjects/MachineLearning_CA1/chromedriver.exe")
    idriver = webdriver.Chrome(service=service)
    idriver.get(config["base_url"])
    idriver.maximize_window()
    yield idriver
   # idriver.quit()

def log_results_to_file(results, filepath="dashboard_data.json"):
    """Log metrics to a JSON file for dashboard consumption."""
    # Check if the file exists; if not, create it
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    # Merge new results with existing data
    existing_data.update(results)

    # Write back to the file
    with open(filepath, "w") as file:
        json.dump(existing_data, file, indent=4)

@pytest.mark.parametrize("data", load_test_data(r"C:\Users\anike\PycharmProjects\MachineLearning_CA1\data\Data.xlsx", "datasheet"))
def test_register(driver, data):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    logger.info("Starting test for new account registration")
    try:
        db = Kraggle_Repo(driver)
        db.iExtractDatabase(data["email"], data["passcode"], data["Dataset"])
        logger.info("Account logged in completed successfully")
        db.iDownloadDatasetKraggle()
        iMachineModelTrainer(driver, r"C:\Users\anike\PycharmProjects\MachineLearning_CA1\test\heart-disease-dataset\heart.csv")
        root = tk.Tk()
        ha = HeartAttackPredictorApp(root)
        ha.create_widgets()
        ha.generate_random_input()
        ha.make_prediction()
        root.mainloop()
    except Exception as e:
        logger.error(f"Error during test execution: {e}")
        raise

def iMachineModelTrainer(driver, data_path):
    try:
        ml = HeartDiseaseML(data_path)
        logging.info("Step-by-step execution started.")
        ml.load_and_clean_data()
        ml.explore_data()
        ml.preprocess_data()
        ml.split_data()

        # Train models
        logistic_model, knn_model, svm_model = ml.train_models()

        # Evaluate models
        logistic_metrics = ml.evaluate_model(logistic_model, "Logistic Regression")
        knn_metrics = ml.evaluate_model(knn_model, "KNN")
        svm_metrics = ml.evaluate_model(svm_model, "SVM")

        # Log results to JSON file for dashboard
        results = {
                   "Logistic Regression": logistic_metrics,
                   "KNN": knn_metrics,
                   "SVM": svm_metrics
        }
        log_results_to_file(results)
        run_dashboard()
        # Save model
        ml.save_model(knn_model, "KNN_model.pkl")
        ml.generatePredictionDataSet()
        # Integrate Prediction Functionality (Post Line 90)
        test_dataset = pd.read_csv("test_dataset.csv")
        random_row = test_dataset.sample(1).iloc[0].tolist()  # Pick a random row
        prediction_probability = predict_heart_attack(random_row)  # Use predict_heart_attack function
        print(f"Prediction probability for the random input: {prediction_probability}")
        logging.info("Model training and evaluation completed successfully.")

    except Exception as e:
        logging.error(f"Error during model training and evaluation: {e}")
        raise

def run_dashboard():
    """Run the Flask dashboard app and launch it in the browser."""
    try:
        # Start the Flask app as a subprocess
        process = subprocess.Popen(["python", r"C:\Users\anike\PycharmProjects\MachineLearning_CA1\dashboard\dashboard_app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Open the dashboard in the default web browser
        webbrowser.open("http://127.0.0.1:5000")
        print("Dashboard is running at http://127.0.0.1:5000. Press Ctrl+C to stop.")
        process.communicate()  # Keep the Flask process running
    except Exception as e:
        print(f"Failed to start the dashboard: {e}")
        raise


if __name__ == "__main__":
    # Run pytest for the test cases
    pytest_exit_code = pytest.main([r"C:\Users\anike\PycharmProjects\MachineLearning_CA1\test\TC_01.py"])  # Adjust path if necessary

    # If pytest completes successfully, launch the dashboard
    if pytest_exit_code == 0:
        run_dashboard()
    else:
        print("Test execution failed. Dashboard will not start.")