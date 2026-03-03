import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Any
import logging

logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Valide les données au format 'Long'. 
    Vérifie que chaque patient a bien ses 21 points (0-100dB) pour is_aided=0 et is_aided=1.
    """
    validation_report = {
        "nb_lignes_initiales": len(df),
        "nb_patients_initiaux": len(df['patient_id'].unique()) if 'patient_id' in df.columns else 0,
        "patients_ecartes": 0
    }

    # 1. Nettoyage de base : conversion numérique forcée des scores et intensités
    cols_check = ['recognition_score', 'intensity_db', 'is_aided']
    for col in cols_check:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Supprimer les lignes où le score est invalide (NaN ou hors limites)
    df = df.dropna(subset=cols_check)
    df = df[(df['recognition_score'] >= 0) & (df['recognition_score'] <= 110)] # On accepte un petit dépassement

    # 2. Filtrage des patients incomplets
    # On veut 21 points (0, 5... 100) pour chaque état
    def est_complet(group):
        points_sans = len(group[group['is_aided'] == 0])
        points_avec = len(group[group['is_aided'] == 1])
        return points_sans == 21 and points_avec == 21

    patients_valides = df.groupby('patient_id').filter(est_complet)
    
    validation_report["nb_patients_finaux"] = len(patients_valides['patient_id'].unique())
    validation_report["patients_ecartes"] = validation_report["nb_patients_initiaux"] - validation_report["nb_patients_finaux"]
    
    return patients_valides, validation_report


def prepare_vocal_sequences(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforme le DataFrame validé en séquences (tenseurs) pour le CNN.
    X = Courbe sans appareil (is_aided=0)
    y = Courbe avec appareil (is_aided=1)
    """
    patients = sorted(df['patient_id'].unique())
    X_list = []
    y_list = []

    for p_id in patients:
        p_data = df[df['patient_id'] == p_id]
        
        # Courbe Sans Appareil (Triée par intensité de 0 à 100)
        curve_sans = p_data[p_data['is_aided'] == 0].sort_values('intensity_db')['recognition_score'].values
        
        # Courbe Avec Appareil (Target)
        curve_avec = p_data[p_data['is_aided'] == 1].sort_values('intensity_db')['recognition_score'].values
        
        X_list.append(curve_sans)
        y_list.append(curve_avec)

    # Conversion en float32 pour TensorFlow et ajout de la dimension "canal"
    # X shape: (nb_patients, 21, 1)
    # y shape: (nb_patients, 21)
    X = np.array(X_list).astype('float32').reshape(-1, 21, 1)
    y = np.array(y_list).astype('float32')
    
    return X, y


def split_vocal_data(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float, 
    random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Découpe les matrices NumPy en train/test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test