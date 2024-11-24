import pytest
import time
import json
import os
from selenium import webdriver
from pages.Kraggle_Repo import Kraggle_Repo
from pages.Module_Execution_ML import HeartDiseaseML
from utils.data_loader import load_test_data
from selenium.webdriver.chrome.service import Service
import logging

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

        # Save model
        ml.save_model(knn_model, "KNN_model.pkl")
        logging.info("Model training and evaluation completed successfully.")
    except Exception as e:
        logging.error(f"Error during model training and evaluation: {e}")
        raise
