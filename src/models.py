"""Préparation des jeux d'entraînement et de test, et entraînement des modèles."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import TARGET


def split_data(df:pd.DataFrame, test_size=0.2, random_state=42):
    """Sépare la cible des variables explicatives, puis en jeux d'entraînement et de test."""
    # X : toutes les colonnes sauf la cible
    X = df.drop(columns=TARGET)
    # y : la cible seule
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)