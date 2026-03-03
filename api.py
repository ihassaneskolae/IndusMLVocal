"""
API FastAPI pour le projet de prédiction audiométrique.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import os
import subprocess
import tensorflow as tf
import numpy as np

app = FastAPI(
    title="Audio Prediction API",
    description="API pour la prédiction de gains prothétiques audiométriques",
    version="1.0.0"
)

# Configuration
MODEL_PATH = "data/06_models/audiogram_model.keras"
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


class AudiogramInput(BaseModel):
    """Schéma pour une ligne d'audiogramme."""
    before_exam_125_Hz: Optional[Any] = None
    before_exam_250_Hz: Optional[Any] = None
    before_exam_500_Hz: Optional[Any] = None
    before_exam_1000_Hz: Optional[Any] = None
    before_exam_2000_Hz: Optional[Any] = None
    before_exam_4000_Hz: Optional[Any] = None
    before_exam_8000_Hz: Optional[Any] = None


class PredictionRequest(BaseModel):
    """Requête de prédiction."""
    data: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    predictions: List[Dict[str, float]]
    valid_rows: int
    invalid_rows: List[Dict[str, Any]]


class TrainResponse(BaseModel):
    """Réponse d'entraînement."""
    status: str
    message: str


def load_model():
    """Charge le modèle entraîné."""
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def validate_row(row: Dict[str, Any], idx: int) -> tuple:
    """Valide une ligne de données."""
    errors = []
    valid_values = {}
    
    for col in INPUT_COLUMNS:
        value = row.get(col)
        
        if value is None or (isinstance(value, float) and pd.isna(value)):
            errors.append(f"{col}: valeur manquante")
        elif isinstance(value, str):
            errors.append(f"{col}: valeur non numérique '{value}'")
        else:
            try:
                num_val = float(value)
                if num_val < -20 or num_val > 150:
                    errors.append(f"{col}: hors limites ({num_val})")
                else:
                    valid_values[col] = num_val
            except (ValueError, TypeError):
                errors.append(f"{col}: conversion impossible")
    
    return valid_values if not errors else None, errors


@app.get("/")
async def root():
    """Route par défaut - informations sur l'API."""
    return {
        "name": "Audio Prediction API",
        "version": "1.0.0",
        "description": "API de prédiction de gains prothétiques audiométriques",
        "endpoints": {
            "/": "Cette page d'information",
            "/train": "POST - Entraîner le modèle",
            "/predict": "POST - Faire des prédictions",
            "/health": "GET - Vérifier l'état de l'API"
        },
        "input_format": {
            "columns": INPUT_COLUMNS,
            "value_range": "[-20, 150] dB"
        },
        "output_format": {
            "columns": OUTPUT_COLUMNS
        }
    }


@app.get("/health")
async def health_check():
    """Vérifie l'état de l'API et du modèle."""
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "status": "healthy",
        "model_available": model_loaded
    }


@app.post("/train", response_model=TrainResponse)
async def train():
    """
    Exécute le pipeline d'entraînement complet.
    """
    try:
        # Exécuter le pipeline Kedro
        result = subprocess.run(
            ["kedro", "run", "--pipeline", "train"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de l'entraînement: {result.stderr}"
            )
        
        return TrainResponse(
            status="success",
            message="Modèle entraîné avec succès"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Effectue des prédictions sur les données fournies.
    
    Retourne les prédictions pour les lignes valides et 
    indique les erreurs pour les lignes invalides.
    """
    model = load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Veuillez d'abord entraîner le modèle via /train"
        )
    
    valid_data = []
    valid_indices = []
    invalid_rows = []
    
    for idx, row in enumerate(request.data):
        valid_values, errors = validate_row(row, idx)
        
        if valid_values:
            valid_data.append(valid_values)
            valid_indices.append(idx)
        else:
            invalid_rows.append({
                "row_index": idx,
                "errors": errors
            })
    
    predictions = []
    if valid_data:
        df = pd.DataFrame(valid_data)
        df = df[INPUT_COLUMNS]  # Assurer l'ordre des colonnes
        
        X_array = df.values.astype(np.float32)
        X_cnn = X_array.reshape((X_array.shape[0], X_array.shape[1], 1))
        
        pred_values = model.predict(X_cnn)
        
        for i, idx in enumerate(valid_indices):
            pred_dict = {
                "original_row_index": idx
            }
            for j, col in enumerate(OUTPUT_COLUMNS):
                pred_dict[col] = float(pred_values[i][j])
            predictions.append(pred_dict)
    
    return PredictionResponse(
        predictions=predictions,
        valid_rows=len(valid_data),
        invalid_rows=invalid_rows
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)