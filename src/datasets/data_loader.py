# data_loader.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.features = None
        self.labels = None
        self.scaler = None

    def load_data(self):
        # Load the data from the CSV file
        self.features = self.data.drop(columns=["target"])
        self.labels = self.data["target"]
        print("Data loaded successfully.")

    def split_data(self, test_size=0.2, random_state=42):
        # Split the data into training and testing sets
        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(
            self.features, self.labels, test_size=test_size, random_state=random_state)
        print("Data split into training and testing sets.")

    def get_data(self):
        return self.features, self.labels

    def get_train_data(self):
        return self.features_train, self.labels_train

    def get_test_data(self):
        return self.features_test, self.labels_test

    def data_summary(self):
        # Display a summary of the loaded data
        num_samples, num_features = self.features.shape
        num_classes = len(self.labels.unique())
        print(f"Number of samples: {num_samples}")
        print(f"Number of features: {num_features}")
        print(f"Number of classes: {num_classes}")

    def standardize_data(self):
        # Standarize the data
        self.scaler = StandardScaler()
        self.features_train = self.scaler.fit_transform(self.features_train)
        self.features_test = self.scaler.transform(self.features_test)

if __name__ == "__main__":
    data_loader = DataLoader("your_data.csv")
    data_loader.load_data()
    data_loader.split_data()
    data_loader.data_summary()
    data_loader.standardize_data()
