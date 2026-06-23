import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

from src.models.train import custom_f1_class_1, custom_recall_class_1, train_decision_tree
from src.models.predict import predict_benchmark, predict_decision_tree

def test_custom_metrics_happy_path():
    """
    Tests that the custom scoring functions correctly evaluate Class 1 performance
    """

    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1]) 
    
    f1 = custom_f1_class_1(y_true, y_pred)
    recall = custom_recall_class_1(y_true, y_pred)
    
    assert np.isclose(recall, 2/3), "Recall for class 1 is calculated incorrectly"
    assert np.isclose(f1, 0.8), "F1 for class 1 is calculated incorrectly"

@patch('src.models.train.joblib.dump')
def test_train_decision_tree_mock(mock_joblib_dump):
    """
    Tests the Decision Tree training pipeline with a minimal mock dataset.
    Uses @patch on joblib.dump to prevent writing dummy .pkl files to the artifacts folder.
    """

    mock_X = pd.DataFrame({
        "temp_acum_1m": [10.5, 20.1, 30.0, 40.5, 50.2, 15.0],
        "viento_prom_1m": [5.0, 15.2, 25.1, 35.0, 45.3, 10.0]
    })
    mock_y = pd.Series([0, 0, 1, 1, 1, 0], name="Estado_Malo")

    cv = KFold(n_splits=2)
    
    # Execute actual training logic
    model = train_decision_tree(mock_X, mock_y, cv)
    
    assert model is not None, "The training function returned None"
    assert isinstance(model, DecisionTreeClassifier), "Returned model is not a Decision Tree"
    assert hasattr(model, "predict"), "The returned model does not have a predict method"
    assert mock_joblib_dump.called, "joblib.dump was never called to save the trained artifact"

def test_benchmark_prediction_logic():
    """
    Tests the persistence benchmark
    """
    mock_data = {
        "Campaña_ID": ["Camp_1", "Camp_1", "Camp_1", "Camp_2", "Camp_2"],
        "Estado_Malo": [0, 1, 1, 1, 0]
    }
    df_mock = pd.DataFrame(mock_data)
    
    # Executes benchmark logic
    preds = predict_benchmark(df_mock)
    
    assert len(preds) == 5, "Benchmark returned a different number of rows"
    
    expected_predictions = [0, 0, 1, 0, 1]
    
    assert preds.tolist() == expected_predictions, "Benchmark did not shift values correctly within campaign bounds"

def test_predict_decision_tree_output():
    """
    Tests that the DT prediction wrapper returns correctly shaped arrays
    """
    # Creates a simple fitted model
    mock_model = DecisionTreeClassifier(random_state=42)
    mock_X_train = pd.DataFrame({"f1": [1, 2, 3], "f2": [1, 2, 3]})
    mock_y_train = pd.Series([0, 1, 1])
    mock_model.fit(mock_X_train, mock_y_train)
    
    mock_X_test = pd.DataFrame({"f1": [1, 3], "f2": [1, 3]})

    preds, probs = predict_decision_tree(mock_model, mock_X_test)
    
    assert len(preds) == 2, "Predictions length mismatch"
    assert len(probs) == 2, "Probabilities length mismatch"
    assert all(p in [0, 1] for p in preds), "Predictions contain invalid classes"
    assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities are out of bounds (0-1)"