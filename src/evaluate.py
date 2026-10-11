"""Évaluation des performances des modèles."""

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series):
    """Calcule le R², la MAE et la RMSE d'un modèle sur le jeu de test."""
    # Le modèle prédit les prix des réservations de test
    y_pred = model.predict(X_test)
    return {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }


def compare_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series):
    """Évalue chaque modèle et renvoie un tableau trié du meilleur au moins bon."""
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test)
    return pd.DataFrame(results).T.sort_values("R2", ascending=False)
