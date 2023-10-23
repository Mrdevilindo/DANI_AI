# data_preprocessing.py

from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.features_train = None
        self.features_test = None
        self.categorical_features = None

    def preprocess_data(self, features_train, features_test, categorical_features=None):
        # Standardize the numerical features using StandardScaler
        numerical_features = [feature for feature in features_train.columns if feature not in categorical_features]
        self.features_train[numerical_features] = self.scaler.fit_transform(features_train[numerical_features])
        self.features_test[numerical_features] = self.scaler.transform(features_test[numerical_features])

        # Encode the categorical features using one-hot encoding
        if categorical_features is not None:
            self.categorical_features = categorical_features
            for feature in categorical_features:
                encoded_feature = pd.get_dummies(features_train[feature])
                features_train = pd.concat([features_train, encoded_feature], axis=1)
                features_test = pd.concat([features_test, encoded_feature], axis=1)
                features_train.drop(columns=[feature], inplace=True)
                features_test.drop(columns=[feature], inplace=True)

        self.features_train = features_train.to_numpy()
        self.features_test = features_test.to_numpy()

        print("Data preprocessed successfully.")

    def get_preprocessed_data(self):
        return self.features_train, self.features_test, self.categorical_features

if __name__ == "__main__":
    data_preprocessor = DataPreprocessor()

    # Load features and labels
    features_train, labels_train = load_training_data()  # Implement your data loading function
    features_test, labels_test = load_testing_data()  # Implement your data loading function

    # Preprocess the data
    data_preprocessor.preprocess_data(features_train, features_test, categorical_features=['gender', 'occupation'])

    # Use the preprocessed data for training and testing
    # ...
