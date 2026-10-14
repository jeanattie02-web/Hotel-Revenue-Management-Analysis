import pandas as pd

from src.features import add_features


def make_df():
    """Deux réservations fabriquées à la main."""
    return pd.DataFrame({
        "stays_in_weekend_nights": [1, 0],
        "stays_in_week_nights": [2, 3],
        "arrival_date_month": ["August", "January"],
        "adults": [2, 1],
        "children": [1, 0],
        "babies": [0, 0],
        "adr": [120.0, 80.0],
        "reserved_room_type": ["A", "A"],
        "assigned_room_type": ["A", "D"],
    })


def test_add_features_ajoute_six_colonnes():
    df_brut = make_df()
    df = add_features(df_brut)
    assert len(df.columns) - len(df_brut.columns) == 6


def test_add_features_calcule_les_bonnes_valeurs():
    df = add_features(make_df())
    assert df["total_nights"].tolist() == [3, 3]
    assert df["arrival_month_num"].tolist() == [8, 1]
    assert df["has_children"].tolist() == [1, 0]
    assert df["room_changed"].tolist() == [0, 1]


def test_add_features_ne_modifie_pas_le_tableau_original():
    df_brut = make_df()
    add_features(df_brut)
    assert "total_nights" not in df_brut.columns