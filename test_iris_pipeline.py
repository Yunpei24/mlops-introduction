from iris_pipeline import *

def test_load_data():
    X_train, X_test, y_train, y_test = load_data()
    # Check the types of the returned values
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


def test_train_model():
    X_train, _, y_train, _ = load_data()
    model = train_model(X_train, y_train)
    # Check if the model is an instance of LogisticRegression
    assert isinstance(model, LogisticRegression)
    # Check if the model has been fitted
    assert hasattr(model, "coef_")

def test_evaluate_model():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    # Capture the printed output
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output
    evaluate_model(model, X_test, y_test)
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    # Check if the output contains accuracy information
    assert "Accuracy:" in output   
