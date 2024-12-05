import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageTk


# Function to load the model and predict heart attack risk
def predict_heart_attack(input_values):
    try:
        # Load the trained model
        model = pickle.load(open("KNN_model.pkl", "rb"))

        # Reshape input to match model requirements
        input_array = np.array(input_values[:-1]).reshape(1, -1)
        print(f"{input_array} Input1")
        # Make prediction (probability of heart attack)
        prediction = model.predict_proba(input_array)
        print(f"{prediction} Input2")
        # Return probability for the positive class
        return prediction[0][1]  # Assuming binary classification

    except FileNotFoundError:
        print("Model file not found. Ensure 'model.pkl' exists.")
        return None
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


# GUI Application
class HeartAttackPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Attack Predictor")
        self.root.geometry("500x600")

        self.create_widgets()

    def create_widgets(self):
        # Title Label
        title_label = tk.Label(self.root, text="Heart Attack Predictor", font=("Arial", 16))
        title_label.pack(pady=10)

        # Input Fields
        self.inputs = {}
        input_labels = [
            "Age",
            "Sex (0=Female, 1=Male)",
            "Chest Pain Type",
            "Max Heart Rate Achieved",
            "Exercise induced Angina",
            "ST Depression",
            "ST Slope",
            "Number of Major Vessels",
            "Thalassemia"

        ]

        for i, label_text in enumerate(input_labels):
            label = tk.Label(self.root, text=label_text, font=("Arial", 10))
            label.pack(pady=5)
            entry = ttk.Entry(self.root)
            entry.pack(pady=5)
            self.inputs[label_text] = entry

        # Generate Button
        generate_button = ttk.Button(self.root, text="Generate Random Input", command=self.generate_random_input)
        generate_button.pack(pady=10)

        # Prediction Button
        predict_button = ttk.Button(self.root, text="Predict", command=self.make_prediction)
        predict_button.pack(pady=20)

        # Heart Display
        self.heart_image_healthy = ImageTk.PhotoImage(Image.open(r"C:\Users\anike\PycharmProjects\MachineLearning_CA1\Animated Heartbeat [Gif].gif").resize((100, 100)))
        self.heart_image_dead = ImageTk.PhotoImage(Image.open(r"C:\Users\anike\PycharmProjects\MachineLearning_CA1\Dead.heart.webp").resize((100, 100)))

        self.heart_label = tk.Label(self.root, image=self.heart_image_healthy)
        self.heart_label.pack(pady=20)

        # Status Label
        self.status_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.status_label.pack(pady=10)

    def generate_random_input(self):
        try:
            # Load test dataset
            test_data = pd.read_csv(r"C:\Users\anike\PycharmProjects\MachineLearning_CA1\test\test_dataset.csv")

            # Select a random row
            random_row = test_data.sample(1).iloc[0]

            # Update input fields with random row values
            for i, label in enumerate(self.inputs.keys()):
                self.inputs[label].delete(0, tk.END)
                self.inputs[label].insert(0, random_row[i])

            messagebox.showinfo("Random Input", "Random values populated successfully!")
        except FileNotFoundError:
            messagebox.showerror("Error", "Test dataset file not found!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def make_prediction(self):
        try:
            # Gather input values
            input_values = [float(self.inputs[label].get()) for label in self.inputs]
            print(f"{input_values} Input3")
            probability = predict_heart_attack(input_values)
            print(f"{probability} Input4")
            # Update heart and status based on prediction
            if probability == 1:
                self.heart_label.config(image=self.heart_image_dead)
                self.status_label.config(text=f"High Risk! Probability: {probability:.2f}", fg="red")
            else:
                self.heart_label.config(image=self.heart_image_healthy)
                self.status_label.config(text=f"Low Risk. Probability: {probability:.2f}", fg="green")

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")


# Load heart images for animation (temporary placeholders)
try:
    app_root = tk.Tk()
    app = HeartAttackPredictorApp(app_root)
    app_root.mainloop()
except Exception as e:
    print(f"Error occurred while creating the GUI: {e}")
