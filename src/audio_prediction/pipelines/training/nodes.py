"""
Nodes pour l'entraînement du modèle 
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
import platform
import logging

logger = logging.getLogger(__name__)


def configure_device() -> str:
    """
    Configure TensorFlow to use the best available device.
    Equivalent to the PyTorch cuda/mps/cpu selection.
    Returns the device name being used.
    """
    gpus = tf.config.list_physical_devices('GPU')
    print(tf.config.list_physical_devices()) 
    if gpus:
        # GPU found — could be NVIDIA CUDA on PC or Metal on Mac
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(platform.system())
            device_name = gpus[0].name
            if platform.system() == "Darwin":
                logger.info("Training on device: Apple Metal GPU (%s)", device_name)
            else:
                logger.info("Training on device: CUDA GPU (%s)", device_name)
            return device_name
        except RuntimeError as e:
            logger.warning("GPU configuration failed, falling back to CPU: %s", e)

    logger.info("Training on device: CPU")
    return "CPU"

def create_model(input_shape, units=128, activation='relu', l2_value=0.01, dropout_rate=None, learning_rate=1e-3):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(filters=32, kernel_size=3, activation=activation, padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation=activation, padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_value))(x)
    
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(7, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=['mae']
    )
    
    return model


def prepare_data_for_cnn(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X_array = X.values.astype(np.float32)
    y_array = y.values.astype(np.float32)
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
    X_train_cnn, y_train_cnn = prepare_data_for_cnn(X_train, y_train)
    input_shape = (X_train_cnn.shape[1], 1)
    device_name = configure_device()
    # Log params to an externally-managed run if one exists
    if mlflow.active_run():
        mlflow.log_params({
            "units": units,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "input_shape": str(input_shape),
            "model_type": "CNN_1D",
            "device": device_name,
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

    # Log training metrics
    if mlflow.active_run():
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

        X_sample = X_train_cnn[:5]
        y_sample = model.predict(X_sample)
        signature = infer_signature(X_sample, y_sample)
        mlflow.tensorflow.log_model(model, "model", signature=signature)

    return model


def evaluate_model(
    model: tf.keras.Model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Dict[str, Any]:
    X_test_cnn, y_test_array = prepare_data_for_cnn(X_test, y_test)
    
    y_pred = model.predict(X_test_cnn)
    
    mse = mean_squared_error(y_test_array, y_pred)
    mae = mean_absolute_error(y_test_array, y_pred)
    r2 = r2_score(y_test_array, y_pred)
    
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
    
    # Log to the same externally-managed run
    if mlflow.active_run():
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