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
