# Import necessary libraries for Irirs dataset pipeline
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data():
    """Function to load and return the Iris dataset as a DataFrame."""
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    X = iris_df.drop('target', axis=1)
    y = iris_df['target']

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    """Function to train and return a Logistic Regression model."""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Function to evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred)*100, "%")