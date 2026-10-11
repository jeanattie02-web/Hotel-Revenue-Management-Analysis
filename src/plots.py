"""Graphiques du projet."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_correlations(df: pd.DataFrame):
    """Trace la matrice de corrélation des variables numériques."""
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matrice de corrélation des variables numériques")
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=15):
    """Trace les top_n variables les plus importantes d'un modèle à base d'arbres."""
    # L'importance de chaque variable, apprise par le modèle pendant .fit
    importances = pd.Series(model.feature_importances_, index=feature_names)
    # On garde les top_n plus importantes
    top = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top.values, color="steelblue")
    ax.invert_yaxis()  # la plus importante en haut
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} des variables les plus importantes")
    fig.tight_layout()
    return fig


def plot_predictions(y_test, y_pred, results: pd.DataFrame):
    """Diagnostic : prédictions vs réel, résidus et comparaison des modèles."""
    residuals = y_test - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # Graphique 1 : prédictions vs valeurs réelles
    axes[0, 0].scatter(y_test, y_pred, alpha=0.2, s=5)
    limites = [y_test.min(), y_test.max()]
    axes[0, 0].plot(limites, limites, "r--", label="Prédiction parfaite")
    axes[0, 0].set_xlabel("Prix réel (€)")
    axes[0, 0].set_ylabel("Prix prédit (€)")
    axes[0, 0].set_title("Prédictions vs valeurs réelles")
    axes[0, 0].legend()

    # Graphique 2 : distribution des résidus
    axes[1, 0].hist(residuals, bins=50)
    axes[1, 0].axvline(0, color="red", linestyle="--")
    axes[1, 0].set_xlabel("Résidu (€)")
    axes[1, 0].set_title(f"Résidus, moyenne : {residuals.mean():.2f} €")

    # Graphique 3 : comparaison du R² des modèles
    axes[0, 1].barh(results.index, results["R2"])  # trou 9
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_xlabel("R²")
    axes[0, 1].set_title("Comparaison des modèles")

    # Graphique 4 : résidus en fonction des prix prédits
    axes[1, 1].scatter(y_pred, residuals, alpha=0.2, s=5)
    axes[1, 1].axhline(0, color="red", linestyle="--")
    axes[1, 1].set_xlabel("Prix prédit (€)")
    axes[1, 1].set_ylabel("Résidu (€)")
    axes[1, 1].set_title("Résidus en fonction des prix prédits")

    fig.tight_layout()
    return fig
