# Hotel Price Prediction & Revenue Management Analysis (Python)

## Description
Projet de modélisation prédictive réalisé dans le cadre du cursus **IMT-BS / ENSIIE**. Ce logiciel analyse les facteurs influençant le tarif journalier moyen (ADR) de réservations hôtelières et compare plusieurs modèles de Machine Learning pour optimiser le Revenue Management.

## Caractéristiques Techniques
* **Langage** : Python 3.x.
* **Librairies Key** : Pandas (Data manipulation), Matplotlib/Seaborn (Visualisation), Scikit-Learn (Modeling).
* **Dataset** : +100 000 réservations (City Hotel vs Resort Hotel) sur la période 2015-2017.
* **Preprocessing** : Traitement des valeurs manquantes, encodage One-Hot des variables catégorielles et filtrage des outliers.

## Pipeline de Data Science
* **Exploration (EDA)** : Analyse de la saisonnalité et matrice de corrélation pour identifier les variables discriminantes.
* **Modélisation** : Implémentation d'une baseline en Régression Linéaire et montée en puissance vers des modèles non-linéaires.
* **Évaluation** : Analyse fine des résidus et mesure de la performance via R², MAE et RMSE.

## Performances des Modèles
Le passage d'une approche linéaire à une approche par ensemble (Random Forest) a permis une amélioration drastique de la précision :

| Modèle | R² Score | MAE (Erreur Moyenne) |
| :--- | :--- | :--- |
| **Linear Regression** | 25.06% | 29.07 € |
| **Decision Tree** | 72.35% | 15.32 € |
| **Random Forest** | **86.76%** | **9.57 €** |

## Gestion de Projet
* **Méthodologie** : Approche scientifique itérative, du nettoyage des données à l'interprétation métier des résultats.

## Installation et Usage
```bash
# Installation des dépendances
pip install pandas numpy matplotlib seaborn scikit-learn

# Exécution de l'analyse
python main.py
