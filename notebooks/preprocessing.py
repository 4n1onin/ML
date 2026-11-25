"""
Prétraitement du dataset UCI « Default of Credit Card Clients ».
Implémente la section 6 du README (Data Preprocessing).

Usage :
    python src/preprocessing.py data/default_of_credit_card_clients.xls
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
TARGET = "DEFAULT"

MONTHS = range(1, 7)  # 1 = septembre 2005 ... 6 = avril 2005
PAY_COLS = [f"PAY_{i}" for i in MONTHS]
BILL_COLS = [f"BILL_AMT{i}" for i in MONTHS]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in MONTHS]


# ---------------------------------------------------------------------------
# 1. Chargement
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Le fichier UCI a une double ligne d'en-tête (X1..X23 puis les vrais noms)."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path, header=1)
    else:
        df = pd.read_excel(path, header=1)  # nécessite xlrd pour .xls
    return df


# ---------------------------------------------------------------------------
# 2. Nettoyage
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Noms cohérents : PAY_0 -> PAY_1 (pour aligner avec BILL_AMT1 / PAY_AMT1)
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": TARGET})

    # ID n'a aucun pouvoir prédictif
    df = df.drop(columns="ID")

    # Doublons exacts : retirés pour éviter qu'une même ligne se retrouve
    # à la fois dans le train et dans le test (fuite de données)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[clean] doublons supprimés : {n_before - len(df)}")

    # Modalités non documentées -> regroupées dans « autres »
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # PAY_x : -2 (pas de consommation), -1 (payé à temps), 0 (crédit renouvelable)
    # et 1..8 (mois de retard). On garde l'échelle ordinale telle quelle.
    # BILL_AMT négatifs = solde créditeur (trop-perçu) -> valeurs légitimes, conservées.
    return df


# ---------------------------------------------------------------------------
# 3. Feature engineering (variables métier simples)
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    limit = df["LIMIT_BAL"]

    # Taux d'utilisation de la ligne de crédit
    util = df[BILL_COLS].div(limit, axis=0)
    df["UTIL_MEAN"] = util.mean(axis=1)
    df["UTIL_LAST"] = util["BILL_AMT1"]

    # Part de la facture précédente effectivement remboursée
    # (PAY_AMTi rembourse la facture BILL_AMT(i+1))
    ratios = []
    for i in range(1, 6):
        bill = df[f"BILL_AMT{i+1}"].clip(lower=0)
        ratios.append(np.where(bill > 0, df[f"PAY_AMT{i}"] / bill.replace(0, np.nan), 1.0))
    df["PAY_RATIO_MEAN"] = np.clip(np.nanmean(np.column_stack(ratios), axis=1), 0, 1)

    # Comportement de retard
    delays = df[PAY_COLS].clip(lower=0)
    df["N_MONTHS_LATE"] = (delays > 0).sum(axis=1)
    df["MAX_DELAY"] = delays.max(axis=1)
    return df


# ---------------------------------------------------------------------------
# 4. Split + transformations (fit sur le train uniquement)
# ---------------------------------------------------------------------------
NOMINAL = ["SEX", "EDUCATION", "MARRIAGE"]
SKEWED = BILL_COLS + PAYAMT_COLS + ["LIMIT_BAL"]


def signed_log1p(x):
    """log1p qui conserve le signe (BILL_AMT peut être négatif)."""
    return np.sign(x) * np.log1p(np.abs(x))


def build_preprocessor(numeric_cols, nominal=NOMINAL):
    """Pour la régression logistique et le MLP (sensibles à l'échelle).
    Les modèles à arbres (RF, XGBoost) peuvent utiliser les données brutes."""
    other_num = [c for c in numeric_cols if c not in SKEWED]
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), list(nominal)),
        ("skew", Pipeline([
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), SKEWED),
        ("num", StandardScaler(), other_num),
    ])


def split(df: pd.DataFrame):
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def imbalance_info(y_train):
    neg, pos = np.bincount(y_train)
    return {"taux_defaut": pos / (neg + pos), "scale_pos_weight": neg / pos}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/default_of_credit_card_clients.xls"
    df = add_features(clean_data(load_data(path)))
    X_train, X_test, y_train, y_test = split(df)

    numeric_cols = [c for c in X_train.columns if c not in NOMINAL]
    pre = build_preprocessor(numeric_cols)
    X_train_scaled = pre.fit_transform(X_train)   # fit sur le train seulement
    X_test_scaled = pre.transform(X_test)

    print(f"[split] train {X_train.shape}, test {X_test.shape}")
    print(f"[split] taux de défaut train {y_train.mean():.3f} | test {y_test.mean():.3f}")
    print(f"[scaled] train {X_train_scaled.shape}")
    print(f"[imbalance] {imbalance_info(y_train)}")
