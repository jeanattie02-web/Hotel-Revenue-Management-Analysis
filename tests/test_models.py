import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.models import load_model, save_model


def test_save_et_load_redonnent_le_meme_modele(tmp_path):
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    model = DecisionTreeRegressor(random_state=42).fit(X, y)

    chemin = tmp_path / "sous_dossier" / "modele.joblib"
    save_model(model, X.columns, chemin)
    model_charge, features = load_model(chemin)

    assert chemin.exists()
    assert features == list(X.columns)
    assert (model_charge.predict(X) == model.predict(X)).all()