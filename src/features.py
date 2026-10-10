import pandas as pd

"""Création des variables dérivées."""

MOIS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def add_features(df):
    """Ajoute les variables dérivées et renvoie une copie."""
    df = df.copy()

    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["arrival_month_num"] = df["arrival_date_month"].map(MOIS)

    # Nombre total de clients : adultes, enfants et bébés
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    # Revenu du séjour : tarif d'une nuit multiplié par le nombre de nuits
    df["total_revenue"] = df["adr"] * df["total_nights"]

    # 1 s'il y a au moins un enfant, 0 sinon
    df["has_children"] = (df["children"] > 0).astype(int)

    # 1 si la chambre attribuée diffère de la chambre réservée, 0 sinon
    df["room_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(
        int
    )

    return df



CATEGORICAL_FEATURES = [
    "hotel", "meal", "market_segment", "distribution_channel",
    "reserved_room_type", "assigned_room_type", "deposit_type",
    "customer_type", "arrival_date_month",
]

NUMERIC_FEATURES = [
    "lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
    "required_car_parking_spaces", "total_of_special_requests", "total_nights",
    "total_guests", "arrival_month_num", "has_children", "room_changed",
    "arrival_date_year",
]

TARGET = "adr"


def encode_features(df: pd.DataFrame):
    """Garde les variables du modèle et encode les catégorielles en 0/1."""
    # On ne garde que les colonnes utiles : numériques, catégorielles et la cible
    df= df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]]

    # On transforme chaque variable catégorielle en colonnes de 0 et de 1
    return pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

def filter_target(df: pd.DataFrame, quantile=0.99):
    """Garde les réservations au tarif strictement positif et sous le quantile choisi."""
    seuil = df[TARGET].quantile(quantile)
    return df[(df[TARGET] > 0) & (df[TARGET] <= seuil)]