import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Any, Tuple
import mlflow
import mlflow.tensorflow
import platform
import logging

logger = logging.getLogger(__name__)

def configure_device() -> str:
    """Configure TensorFlow pour le GPU du Mac M1."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if platform.system() == "Darwin":
                return "MPS"
        except Exception:
            pass
    return "CPU"

def create_vocal_model(input_shape=(21, 1), learning_rate=1e-3, units=128, dropout_rate=0.2):
    """
    Architecture CNN pour prédire l'amélioration vocale.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2, padding='same'),
        
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.Flatten(),
        
        # Utilisation du paramètre units
        layers.Dense(units, activation='relu'),
        # Utilisation du paramètre dropout_rate
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu'),
        
        layers.Dense(21, activation='linear') 
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=['mae']
    )
    return model

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    units: int,            # Ajouté pour correspondre au pipeline
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout_rate: float     # Ajouté pour correspondre au pipeline
) -> tf.keras.Model:
    
    configure_device()
    mlflow.tensorflow.autolog(disable=True)

    if len(X_train.shape) == 2:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # On passe tous les paramètres au créateur du modèle
    model = create_vocal_model(
        input_shape=(21, 1), 
        learning_rate=learning_rate,
        units=units,
        dropout_rate=dropout_rate
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    if mlflow.active_run():
        mlflow.tensorflow.log_model(model, "modele_vocal_uniquement")

    return model

def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    
    if len(X_test.shape) == 2:
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Évaluation Vocale Terminée - MAE: {mae:.2f}% de précision sur la courbe")
    return {"test_mse": float(mse), "test_mae": float(mae)}