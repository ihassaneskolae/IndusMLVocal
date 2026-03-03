"""
Tests pour les nodes du projet.
"""
import pytest
import pandas as pd
import numpy as np
from audio_prediction.pipelines.data_processing.nodes import (
    validate_data, split_features_target, split_train_test
)
from audio_prediction.pipelines.training.nodes import train_model, evaluate_model
from audio_prediction.pipelines.inference.nodes import validate_prediction_input, predict


INPUT_COLUMNS = [
    "before_exam_125_Hz", "before_exam_250_Hz", "before_exam_500_Hz",
    "before_exam_1000_Hz", "before_exam_2000_Hz", "before_exam_4000_Hz",
    "before_exam_8000_Hz"
]
OUTPUT_COLUMNS = [
    "after_exam_125_Hz", "after_exam_250_Hz", "after_exam_500_Hz",
    "after_exam_1000_Hz", "after_exam_2000_Hz", "after_exam_4000_Hz",
    "after_exam_8000_Hz"
]


@pytest.fixture
def sample_valid_data():
    """Génère des données valides de test."""
    np.random.seed(42)
    n_samples = 100
    data = {}
    for col in INPUT_COLUMNS:
        data[col] = np.random.randint(0, 100, n_samples)
    for col in OUTPUT_COLUMNS:
        data[col] = np.random.randint(0, 80, n_samples)
    return pd.DataFrame(data)


@pytest.fixture
def sample_invalid_data():
    """Génère des données avec des valeurs invalides."""
    data = {
        "before_exam_125_Hz": [50, np.nan, 30, "A", 40],
        "before_exam_250_Hz": [45, 50, np.nan, 35, 200],
        "before_exam_500_Hz": [40, 45, 35, 30, 35],
        "before_exam_1000_Hz": [35, 40, 30, 25, 30],
        "before_exam_2000_Hz": [30, 35, 25, 20, 25],
        "before_exam_4000_Hz": [25, 30, 20, 15, 20],
        "before_exam_8000_Hz": [20, 25, 15, 10, 15],
        "after_exam_125_Hz": [30, 35, 20, 25, 25],
        "after_exam_250_Hz": [25, 30, 15, 20, 20],
        "after_exam_500_Hz": [20, 25, 10, 15, 15],
        "after_exam_1000_Hz": [15, 20, 5, 10, 10],
        "after_exam_2000_Hz": [10, 15, 0, 5, 5],
        "after_exam_4000_Hz": [5, 10, 0, 0, 0],
        "after_exam_8000_Hz": [0, 5, 0, 0, 0],
    }
    return pd.DataFrame(data)


class TestDataProcessingNodes:
    """Tests pour les nodes de data processing."""
    
    def test_validate_data_valid(self, sample_valid_data):
        """Test validation avec données valides."""
        cleaned_df, report = validate_data(sample_valid_data, INPUT_COLUMNS, OUTPUT_COLUMNS)
        
        assert len(cleaned_df) == len(sample_valid_data)
        assert report["valid_rows"] == len(sample_valid_data)
        assert report["removed_rows"] == 0
    
    def test_validate_data_invalid(self, sample_invalid_data):
        """Test validation avec données invalides."""
        cleaned_df, report = validate_data(sample_invalid_data, INPUT_COLUMNS, OUTPUT_COLUMNS)
        
        assert len(cleaned_df) < len(sample_invalid_data)
        assert len(report["invalid_rows"]) > 0
        assert report["removed_rows"] > 0
    
    def test_validate_data_missing_columns(self, sample_valid_data):
        """Test validation avec colonnes manquantes."""
        df_missing = sample_valid_data.drop(columns=["before_exam_125_Hz"])
        
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            validate_data(df_missing, INPUT_COLUMNS, OUTPUT_COLUMNS)
    
    def test_split_features_target(self, sample_valid_data):
        """Test séparation features/target."""
        X, y = split_features_target(sample_valid_data, INPUT_COLUMNS, OUTPUT_COLUMNS)
        
        assert list(X.columns) == INPUT_COLUMNS
        assert list(y.columns) == OUTPUT_COLUMNS
        assert len(X) == len(y)
    
    def test_split_train_test(self, sample_valid_data):
        """Test split train/test."""
        X, y = split_features_target(sample_valid_data, INPUT_COLUMNS, OUTPUT_COLUMNS)
        X_train, X_test, y_train, y_test = split_train_test(X, y, 0.2, 42)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


class TestTrainingNodes:
    """Tests pour les nodes d'entraînement."""
    
    def test_train_model(self, sample_valid_data):
        """Test entraînement du modèle."""
        X, y = split_features_target(sample_valid_data, INPUT_COLUMNS, OUTPUT_COLUMNS)
        X_train, X_test, y_train, y_test = split_train_test(X, y, 0.2, 42)
        
        model = train_model(X_train, y_train, units=10, epochs=2, batch_size=32)
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_evaluate_model(self, sample_valid_data):
        """Test évaluation du modèle."""
        X, y = split_features_target(sample_valid_data, INPUT_COLUMNS, OUTPUT_COLUMNS)
        X_train, X_test, y_train, y_test = split_train_test(X, y, 0.2, 42)
        
        model = train_model(X_train, y_train, units=10, epochs=2, batch_size=32)
        metrics = evaluate_model(model, X_test, y_test)
        
        assert "overall" in metrics
        assert "mse" in metrics["overall"]
        assert "mae" in metrics["overall"]
        assert "r2" in metrics["overall"]
        assert "per_frequency" in metrics


class TestInferenceNodes:
    """Tests pour les nodes d'inférence."""
    
    def test_validate_prediction_input_valid(self, sample_valid_data):
        """Test validation entrée prédiction valide."""
        input_df = sample_valid_data[INPUT_COLUMNS].head(10)
        valid_df, errors = validate_prediction_input(input_df, INPUT_COLUMNS)
        
        assert len(valid_df) == 10
        assert len(errors) == 0
    
    def test_validate_prediction_input_invalid(self, sample_invalid_data):
        """Test validation entrée prédiction invalide."""
        input_df = sample_invalid_data[INPUT_COLUMNS]
        valid_df, errors = validate_prediction_input(input_df, INPUT_COLUMNS)
        
        assert len(errors) > 0
        # Vérifier que les erreurs contiennent les indices des lignes
        for error in errors:
            assert "row_index" in error
            assert "errors" in error
    
    def test_predict(self, sample_valid_data):
        """Test prédiction."""
        X, y = split_features_target(sample_valid_data, INPUT_COLUMNS, OUTPUT_COLUMNS)
        X_train, X_test, y_train, y_test = split_train_test(X, y, 0.2, 42)
        
        model = train_model(X_train, y_train, units=10, epochs=2, batch_size=32)
        predictions = predict(model, X_test, OUTPUT_COLUMNS)
        
        assert len(predictions) == len(X_test)
        assert list(predictions.columns) == OUTPUT_COLUMNS


class TestAPIValidation:
    """Tests de validation pour l'API."""
    
    def test_row_validation_nan(self):
        """Test détection des NaN."""
        from audio_prediction.pipelines.inference.nodes import validate_prediction_input
        
        data = pd.DataFrame([{
            "before_exam_125_Hz": np.nan,
            "before_exam_250_Hz": 50,
            "before_exam_500_Hz": 45,
            "before_exam_1000_Hz": 40,
            "before_exam_2000_Hz": 35,
            "before_exam_4000_Hz": 30,
            "before_exam_8000_Hz": 25,
        }])
        
        valid_df, errors = validate_prediction_input(data, INPUT_COLUMNS)
        
        assert len(errors) == 1
        assert "valeur manquante" in errors[0]["errors"][0]
    
    def test_row_validation_string(self):
        """Test détection des strings."""
        from audio_prediction.pipelines.inference.nodes import validate_prediction_input
        
        data = pd.DataFrame([{
            "before_exam_125_Hz": "ABC",
            "before_exam_250_Hz": 50,
            "before_exam_500_Hz": 45,
            "before_exam_1000_Hz": 40,
            "before_exam_2000_Hz": 35,
            "before_exam_4000_Hz": 30,
            "before_exam_8000_Hz": 25,
        }])
        
        valid_df, errors = validate_prediction_input(data, INPUT_COLUMNS)
        
        assert len(errors) == 1
        assert "non numérique" in errors[0]["errors"][0]