"""
Nodes pour l'entraînement du modèle - Utilise le modèle CNN officiel du cours.
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple
import mlflow
import mlflow.tensorflow


def create_model(input_shape, units=128, activation='relu', l2_value=0.01, dropout_rate=None, learning_rate=1e-3):
    """
    Crée le modèle CNN - Fonction officielle du cours.
    
    Args:
        input_shape: Format (dim, 1) pour Conv1D
        units: Nombre de neurones dans la couche dense
        activation: Fonction d'activation
        l2_value: Valeur de régularisation L2
        dropout_rate: Taux de dropout (None pour désactiver)
        learning_rate: Taux d'apprentissage
    
    Returns:
        Modèle Keras compilé
    """
    inputs = layers.Input(shape=input_shape)

    # Couches de convolution
    x = layers.Conv1D(filters=32, kernel_size=3, activation=activation, padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation=activation, padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    # Aplatir
    x = layers.Flatten()(x)

    # Couches denses
    x = layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_value))(x)
    
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    # Sortie: 7 valeurs (une par fréquence)
    outputs = layers.Dense(7, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=['mae']
    )
    
    return model


def prepare_data_for_cnn(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prépare les données pour le modèle CNN (reshape pour Conv1D).
    
    Args:
        X: Features DataFrame
        y: Targets DataFrame
    
    Returns:
        Tuple (X_reshaped, y_array)
    """
    X_array = X.values.astype(np.float32)
    y_array = y.values.astype(np.float32)
    
    # Reshape pour Conv1D: (samples, timesteps, features) -> (samples, 7, 1)
    X_reshaped = X_array.reshape((X_array.shape[0], X_array.shape[1], 1))
    
    return X_reshaped, y_array


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    units: int = 128,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    Entraîne le modèle CNN.
    
    Args:
        X_train, y_train: Données d'entraînement
        units: Neurones dans la couche dense
        epochs: Nombre d'époques
        batch_size: Taille du batch
        learning_rate: Taux d'apprentissage
        dropout_rate: Taux de dropout
    
    Returns:
        Modèle entraîné
    """
    # Préparer les données
    X_train_cnn, y_train_cnn = prepare_data_for_cnn(X_train, y_train)
    
    # Créer le modèle
    input_shape = (X_train_cnn.shape[1], 1)  # (7, 1)
    with mlflow.start_run() as run:
        # Log des hyperparamètres
        mlflow.log_params({
            "units": units,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "input_shape": str(input_shape),
        })

        model = create_model(
            input_shape=input_shape,
            units=units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = model.fit(
            X_train_cnn, y_train_cnn,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )

        # Log des métriques d'entraînement
        for epoch_idx, (loss, mae) in enumerate(
            zip(history.history['loss'], history.history['mae'])
        ):
            mlflow.log_metric("train_loss", loss, step=epoch_idx)
            mlflow.log_metric("train_mae", mae, step=epoch_idx)

        if 'val_loss' in history.history:
            for epoch_idx, (val_loss, val_mae) in enumerate(
                zip(history.history['val_loss'], history.history['val_mae'])
            ):
                mlflow.log_metric("val_loss", val_loss, step=epoch_idx)
                mlflow.log_metric("val_mae", val_mae, step=epoch_idx)

        # Log du modèle
        mlflow.tensorflow.log_model(model, "model")

        print(f"Run ID: {run.info.run_id}")

    return model


def evaluate_model(
    model: tf.keras.Model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Dict[str, Any]:
    """
    Évalue les performances du modèle.
    """
    X_test_cnn, y_test_array = prepare_data_for_cnn(X_test, y_test)
    
    y_pred = model.predict(X_test_cnn)
    
    mse = mean_squared_error(y_test_array, y_pred)
    mae = mean_absolute_error(y_test_array, y_pred)
    r2 = r2_score(y_test_array, y_pred)
    
    # Métriques par fréquence
    per_frequency_metrics = {}
    columns = y_test.columns.tolist()
    for i, col in enumerate(columns):
        per_frequency_metrics[col] = {
            "mse": float(mean_squared_error(y_test_array[:, i], y_pred[:, i])),
            "mae": float(mean_absolute_error(y_test_array[:, i], y_pred[:, i])),
            "r2": float(r2_score(y_test_array[:, i], y_pred[:, i]))
        }
    
    metrics = {
        "overall": {"mse": float(mse), "mae": float(mae), "r2": float(r2)},
        "per_frequency": per_frequency_metrics
    }
    
    # Log eval metrics into the most recent run of the same experiment
    mlflow.set_experiment("audio_prediction")
    last_run = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    if not last_run.empty:
        run_id = last_run.iloc[0]["run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                "test_mse": float(mse),
                "test_mae": float(mae),
                "test_r2": float(r2),
            })
            for col, freq_metrics in per_frequency_metrics.items():
                freq_label = col.replace("after_exam_", "").replace("_Hz", "")
                mlflow.log_metric(f"mse_{freq_label}", freq_metrics["mse"])
                mlflow.log_metric(f"mae_{freq_label}", freq_metrics["mae"])
                mlflow.log_metric(f"r2_{freq_label}", freq_metrics["r2"])

    print(f"Model Performance - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return metrics