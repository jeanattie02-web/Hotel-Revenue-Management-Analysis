# 🏨 Hotel Price Prediction & Revenue Management

> **Modélisation prédictive du tarif journalier moyen (ADR) sur plus de 100 000 réservations hôtelières — du nettoyage des données à l'optimisation du Revenue Management.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-11557C)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## 📌 Contexte

Projet réalisé dans le cadre du **cursus Master 1 à IMT-BS (cours INF 4002)**.

L'objectif est de **comprendre les leviers tarifaires** d'un groupe hôtelier (City Hotel vs Resort Hotel) et de **prédire le prix journalier moyen** (Average Daily Rate – ADR) à partir des caractéristiques d'une réservation. Au-delà de la performance statistique, l'analyse vise à fournir des **recommandations métier actionnables** pour le Revenue Management.

**Auteurs** : Jean ATTIE · Léo GAUTIER · Antoine DEMANGHON

---

## 🎯 Problématique

> *Quels sont les facteurs qui déterminent le prix d'une nuit d'hôtel, et peut-on le prédire de manière fiable pour optimiser la stratégie tarifaire ?*

Trois sous-questions structurent l'analyse :
1. Quelles variables influencent le plus l'ADR (saisonnalité, type d'hôtel, segment de clientèle) ?
2. Un modèle linéaire suffit-il, ou faut-il capter des relations non-linéaires ?
3. Quelle est la précision réellement atteignable, et quel est le coût d'erreur business ?

---

## 🗂️ Dataset

| Caractéristique | Valeur |
|---|---|
| **Volume** | ~119 000 réservations |
| **Période** | 2015 – 2017 |
| **Types d'hôtels** | City Hotel & Resort Hotel |
| **Variable cible** | `adr` (Average Daily Rate, en €) |
| **Source** | [Hotel Booking Demand Dataset – Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) |

---

## ⚙️ Stack Technique

| Domaine | Outils |
|---|---|
| **Langage** | Python 3.x |
| **Manipulation de données** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn (LinearRegression, DecisionTreeRegressor, RandomForestRegressor) |
| **Environnement** | Jupyter Notebook / Google Colab |

---

## 🔬 Pipeline de Data Science

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  1. Cleaning │ → │   2. EDA     │ → │ 3. Feature   │ → │ 4. Modeling  │ → │ 5. Evaluation│
│  & Preproc.  │   │  Saisonnalité│   │  Engineering │   │  3 modèles   │   │ R² / MAE /   │
│              │   │  Corrélations│   │  One-Hot enc.│   │  comparés    │   │  RMSE        │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

### 1. Préprocessing
- Imputation des valeurs manquantes (`children`, `country`, `agent`, `company`)
- Suppression des bookings incohérents (0 adulte/enfant/bébé, ADR négatif ou aberrant)
- **One-Hot Encoding** des variables catégorielles (`hotel`, `meal`, `market_segment`, `reserved_room_type`…)
- Création de variables dérivées : `total_nights`, `total_guests`, `has_children`, `room_changed`

### 2. Analyse exploratoire (EDA)
- Étude de la **saisonnalité** mensuelle de l'ADR
- **Matrice de corrélation** sur les variables numériques pour identifier les prédicteurs forts et la multicolinéarité
- Détection visuelle des outliers via boxplots

### 3. Modélisation
Trois modèles entraînés sur le même split train/test pour une comparaison équitable :
- **Linear Regression** — baseline interprétable
- **Decision Tree** — capture des non-linéarités
- **Random Forest** — réduction de la variance par bagging

---

## 📊 Résultats

### Performance comparée des modèles

| Modèle | R² Score | MAE | Lecture |
|---|:---:|:---:|---|
| Linear Regression | 25,06 % | 29,07 € | ❌ Insuffisant — relations non-linéaires non captées |
| Decision Tree | 72,35 % | 15,32 € | ⚠️ Bon mais variance élevée (overfitting) |
| **🏆 Random Forest** | **86,76 %** | **9,57 €** | ✅ Modèle retenu — erreur moyenne < 10 € |

> Le passage d'une régression linéaire à un Random Forest **divise l'erreur moyenne par 3** et fait passer le R² de 25 % à près de 87 %.

### Variables les plus prédictives (Random Forest)

| Rang | Variable | Importance |
|:---:|---|:---:|
| 🥇 | `total_guests` | 20,1 % |
| 🥈 | `arrival_month_num` | 18,6 % |
| 🥉 | `hotel_Resort Hotel` | 11,4 % |
| 4 | `arrival_date_week_number` | 6,9 % |
| 5 | `lead_time` | 6,7 % |
| 6 | `market_segment_Online TA` | 6,1 % |

> 📌 **Insight métier** : le tarif est piloté avant tout par **la composition du groupe** et **la période de l'année**, deux leviers directement actionnables en Revenue Management.

---

## 📈 Visualisations clés

Le projet produit trois visualisations majeures, disponibles dans `/figures` :

| Fichier | Description |
|---|---|
| `01_correlations_variables.png` | Matrice de corrélation des variables numériques |
| `02_importance_features_rf.png` | Top 15 des variables les plus prédictives (Random Forest) |
| `03_performance_modele_rf.png` | Diagnostic complet : prédictions vs réel, distribution des résidus, comparaison des modèles |

---

## 💡 Conclusions & enseignements

✅ **Un modèle non-linéaire est indispensable** : la relation entre les caractéristiques d'une réservation et son prix est trop complexe pour une régression linéaire.

✅ **Le Random Forest atteint une précision exploitable** (MAE ≈ 9,57 €), suffisante pour alimenter un outil d'aide à la décision tarifaire.

✅ **Trois leviers business émergent** : la taille du groupe, la saisonnalité, et le type d'établissement.

⚠️ **Limites identifiées** :
- Le dataset s'arrête en 2017 — un réentraînement est nécessaire pour intégrer la dynamique post-COVID.
- Certains résidus extrêmes suggèrent l'existence de tarifs "premium" non capturés (suites, packages spéciaux).
- Un Gradient Boosting (XGBoost / LightGBM) pourrait être testé en extension.

---

## 🚀 Installation & Utilisation

### Prérequis
- Python ≥ 3.8
- pip

### Installation

```bash
# Cloner le repository
git clone https://github.com/<votre-user>/hotel-price-prediction.git
cd hotel-price-prediction

# Installer les dépendances
pip install -r requirements.txt
```

### Contenu de `requirements.txt`
```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

### Exécution

```bash
# Option 1 — Notebook interactif (recommandé)
jupyter notebook Hotel-Booking.ipynb

# Option 2 — Script Python
python main.py
```

---

## 📁 Structure du projet

```
hotel-price-prediction/
├── data/
│   └── hotel_bookings.csv
├── figures/
│   ├── 01_correlations_variables.png
│   ├── 02_importance_features_rf.png
│   └── 03_performance_modele_rf.png
├── notebooks/
│   └── Hotel-Booking.ipynb
├── main.py
├── requirements.txt
└── README.md
```

---

## 🧭 Méthodologie

Approche **scientifique itérative** : chaque étape (cleaning → EDA → modélisation → évaluation) a été validée avant de passer à la suivante, avec retour en arrière si nécessaire. L'interprétation métier des résultats est intégrée à chaque palier plutôt que reportée à la fin.

---

## 👥 Auteurs

| Nom | Contribution |
|---|---|
| **Jean ATTIE** | Data cleaning · EDA · Modélisation |
| **Léo GAUTIER** | Feature engineering · Évaluation · Visualisations |
| **Antoine DEMANGHON** | Préprocessing · Modélisation · Documentation |

---

## 📜 Licence

Projet académique réalisé dans le cadre du cours **INF 4002 – IMT-BS (Master 1)**.
Usage pédagogique uniquement.

---

<p align="center">
  <i>Made with 🐍 Python at IMT Business School · 2025</i>
</p>
