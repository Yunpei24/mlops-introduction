# Import necessary libraries for Irirs dataset pipeline
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

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
    acc = accuracy_score(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy:", acc * 100, "%")
    return acc

# Visualize Data and Model Performance
def plot_confusion_matrix(y_test, y_pred):
    """Function to plot the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(set(y_test)))
    plt.xticks(tick_marks, set(y_test))
    plt.yticks(tick_marks, set(y_test))

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    acc = evaluate_model(model, X_test, y_test)
    
    # Plot confusion matrix
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)