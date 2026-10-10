"""Chargement et nettoyage des données de réservations hôtelières."""

import pandas as pd


def load_data(path):
    """Charge le CSV des réservations et renvoie un DataFrame."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, adr_max=1000):
    """Nettoie les réservations et renvoie une copie.

    Remplit les valeurs manquantes, retire les réservations sans client
    et les ADR négatifs ou supérieurs à adr_max.
    """
    df = df.copy()

    # Valeurs neutres pour les colonnes qui contiennent des valeurs manquantes
    valeurs_par_defaut = {
        "children": 0,
        "country": "Unknown",
        "agent": 0,
        "company": "Unknown",
    }

    df = df.fillna(value=valeurs_par_defaut)

    # Une réservation doit concerner au moins un client
    nb_clients = df["adults"] + df["children"] + df["babies"]
    df = df[nb_clients > 0]

    # On écarte les tarifs négatifs et les valeurs aberrantes
    df = df[(df["adr"] >= 0) & (df["adr"] <= adr_max)]

    return df
