"""Pipeline complet : des données brutes à la comparaison des modèles."""

from src.data import clean_data, load_data
from src.evaluate import compare_models
from src.features import (
    NUMERIC_FEATURES,
    TARGET,
    add_features,
    encode_features,
    filter_target,
)
from src.models import split_data, train_models
from src.plots import plot_correlations, plot_feature_importance, plot_predictions

DATA_PATH = "data/hotel_booking.csv"


def main():
    # Préparation des données
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = add_features(df)
    df = filter_target(df)
    df = encode_features(df)

    # Modélisation
    X_train, X_test, y_train, y_test = split_data(df)
    models = train_models(X_train, y_train)

    # Évaluation
    results = compare_models(models, X_test, y_test)
    print(results)

    # Graphiques (Sauvegardés dans figures/)
    # Matrice de Corrélation:
    fig = plot_correlations(df[NUMERIC_FEATURES + [TARGET]])
    fig.savefig("figures/01_correlations_variables.png", dpi=150)

    # Indicateurs les plus importants:
    fig = plot_feature_importance(models["Random Forest"], X_train.columns)
    fig.savefig("figures/02_importance_features_rf.png", dpi=150)

    # Prix prédit contre prix réel
    y_pred = models["Random Forest"].predict(X_test)
    fig = plot_predictions(y_test, y_pred, results)
    fig.savefig("figures/03_performance_models_rf.png", dpi=150)


if __name__ == "__main__":
    main()
