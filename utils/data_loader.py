import pandas as pd


def load_test_data(file_path, sheet_name):

    try:
        # Load the data into a DataFrame
        test_data = pd.read_excel(file_path, sheet_name=sheet_name)

        # Return data as a list of dictionaries
        return test_data.to_dict(orient='records')
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at path {file_path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while loading test data: {e}")
