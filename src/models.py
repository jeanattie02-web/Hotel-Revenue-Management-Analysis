"""Préparation des données, entraînement, sauvegarde et chargement des modèles."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.features import TARGET


def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Sépare la cible des variables explicatives, puis en jeux d'entraînement et de test."""
    # X : toutes les colonnes sauf la cible
    X = df.drop(columns=TARGET)
    # y : la cible seule
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_models(X_train: pd.DataFrame, y_train: pd.Series):
    """Entraîne les trois modèles et les renvoie dans un dictionnaire."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10, min_samples_split=20, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }
    # On parcourt les modèles du dictionnaire
    for model in models.values():
        # Puis on entraîne chaque modèle sur les données d'entraînement
        model.fit(X_train, y_train)
    return models


# Sauvegarde du modèle entrainé


def save_model(model, feature_names, path="models/random_forest.joblib"):
    """Enregistre le modèle et la liste des colonnes utilisées à l'entraînement."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": list(feature_names)}, path
    )  # sérialisation


def load_model(path="models/random_forest.joblib"):
    """Recharge le modèle et la liste des colonnes."""
    artefact = joblib.load(path)
    return artefact["model"], artefact["features"]
