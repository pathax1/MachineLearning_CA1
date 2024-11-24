from selenium.webdriver import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from utils.CommonFunctions import iaction, highlight_element
import time
import os
import zipfile
class Kraggle_Repo:
    def __init__(self, driver):
        self.driver = driver  # Use the correct WebDriver instance
        self.SigninBtn = "//span[normalize-space()='Sign In']"
        self.SigninBtnEmail = "//span[normalize-space()='Sign in with Email']"
        self.SigninEmailTxtbox = "//input[@id=':r6:']"
        self.SigninPasscodeTxtbox = "//input[@id=':r7:']"
        self.SigninBtn2 = "//span[normalize-space()='Sign In']"
        self.ValUsernameTitle = "//h1[normalize-space()='Welcome, TestExecutionML!']"
        self.SearchBarClick = "//input[@id=':r1:']"
        self.SearchDatasetBar = "//input[@aria-label='Search on Kaggle']"
        self.SelectDatasetfromSearchResults = "//div[normalize-space()='Heart Disease Dataset']"
        self.DownloadDatasetKraggle = "//span[normalize-space()='Download']"

    def iExtractDatabase(self, email, password, dataset):
        # Click the 'Sign In' button
        iaction(self.driver, "Button", "XPATH", self.SigninBtn)
        time.sleep(5)

        # Click the 'Sign In with email' button
        iaction(self.driver, "Button", "XPATH", self.SigninBtnEmail)
        time.sleep(5)

        # Enter the email address in the email input field
        iaction(self.driver, "Textbox", "XPATH", self.SigninEmailTxtbox, email)

        # Enter the password in the password input field
        iaction(self.driver, "Textbox", "XPATH", self.SigninPasscodeTxtbox, password)

        # Click the 'Submit' button to log in
        iaction(self.driver, "Button", "XPATH", self.SigninBtn2)
        time.sleep(5)

        # Locate the element with the label "Welcome, TestExecutionML!"
        welcome_label = self.driver.find_element(By.XPATH, "//h1[contains(text(), 'Welcome, TestExecutionML!')]")

        # Validate the text
        expected_text = "Welcome, TestExecutionML!"
        actual_text = welcome_label.text

        if actual_text == expected_text:
            print("Validation Passed: The label text is correct.")
            highlight_element(self.driver, welcome_label)
        else:
            print(f"Validation Failed: Expected '{expected_text}', but got '{actual_text}'.")

        # Click the Database SearchBar
        iaction(self.driver, "Button", "XPATH", self.SearchBarClick)
        time.sleep(5)

        # Enter the dataset you want to search
        iaction(self.driver, "Textbox", "XPATH", self.SearchDatasetBar, dataset)

        # Locate the search bar element and simulate pressing 'Enter'
        search_bar = self.driver.find_element("xpath", self.SearchDatasetBar)
        search_bar.send_keys(Keys.RETURN)
        time.sleep(5)

        # Click the Database Link
        iaction(self.driver, "Button", "XPATH", self.SelectDatasetfromSearchResults)
        time.sleep(5)

        # Click the Download button
        iaction(self.driver, "Button", "XPATH", self.DownloadDatasetKraggle)
        time.sleep(5)

    def iDownloadDatasetKraggle(self):
        try:

            # Set the Kaggle token location
            os.environ['KAGGLE_CONFIG_DIR'] = r'C:\Users\anike\PycharmProjects\MachineLearning_CA1\Kraggle_Token'

            # Run the Kaggle CLI command
            download_command = "kaggle datasets download -d johnsmith88/heart-disease-dataset"
            result = os.system(download_command)

            # Check if the command ran successfully
            if result != 0:
                raise Exception("Failed to download dataset. Ensure 'kaggle' is properly installed and configured.")

            # Unzip the dataset
            zip_path = "heart-disease-dataset.zip"
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"{zip_path} not found after download.")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("heart-disease-dataset")
            print("Dataset downloaded and extracted successfully.")

        except Exception as e:
            print(f"Error: {e}")
            raise
