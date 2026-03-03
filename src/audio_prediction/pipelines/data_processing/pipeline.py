from kedro.pipeline import Pipeline, node, pipeline
# On n'importe QUE ce qu'on utilise pour la vocale
from .nodes import (
    validate_data, 
    prepare_vocal_sequences, 
    split_vocal_data
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # 1. Nettoyage des données (Format Long)
        node(
            func=validate_data,
            inputs="raw_vocal_data", 
            outputs=["cleaned_vocal_data", "validation_report"],
            name="validate_vocal_node"
        ),
        # 2. Transformation en séquences pour le CNN (Reshape)
        node(
            func=prepare_vocal_sequences,
            inputs="cleaned_vocal_data",
            outputs=["X_vocal_full", "y_vocal_full"],
            name="prepare_vocal_sequences_node"
        ),
        # 3. Découpage Train/Test (Format NumPy)
        node(
            func=split_vocal_data,
            inputs={
                "X": "X_vocal_full",
                "y": "y_vocal_full",
                "test_size": "params:data_processing.test_size",
                "random_state": "params:data_processing.random_state",
            },
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_vocal_train_test_node"
        )
    ])