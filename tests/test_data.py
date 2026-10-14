import pandas as pd

from src.data import clean_data


def make_df():
    """Trois réservations fabriquées à la main."""
    return pd.DataFrame({
        "adults": [2, 0, 1],
        "children": [None, 0, 1],
        "babies": [0, 0, 0],
        "country": ["FRA", None, "PRT"],
        "agent": [9.0, None, None],
        "company": [None, None, 40.0],
        "adr": [100.0, 80.0, -5.0],
    })


def test_clean_data_remplit_les_valeurs_manquantes():
    df = clean_data(make_df())
    assert df[["children", "country", "agent", "company"]].isna().sum().sum() == 0


def test_clean_data_garde_les_bonnes_reservations():
    df = clean_data(make_df())
    assert len(df) == 1


def test_clean_data_ne_modifie_pas_le_tableau_original():
    df_brut = make_df()
    clean_data(df_brut)
    assert df_brut["children"].isna().sum() == 1