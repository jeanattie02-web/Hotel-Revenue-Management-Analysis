"""Pipeline complet : des données brutes à la comparaison des modèles."""

from src.data import clean_data, load_data
from src.evaluate import compare_models
from src.features import add_features, encode_features, filter_target
from src.models import split_data, train_models

DATA_PATH = "database/hotel_booking.csv"


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


if __name__ == "__main__":
    main()