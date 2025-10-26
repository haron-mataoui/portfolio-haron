# pages/Loan_Default_Classification.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuration ---
st.set_page_config(page_title="Loan Default - Analyse & Démo", page_icon="💰", layout="wide")
st.title(" Prédiction du statut de prêt — Analyse complète & démonstration interactive")

# --- Chargement du CSV ---
@st.cache_data
def load_data(csv_name="loan_data.csv"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier '{csv_name}' est introuvable dans {current_dir}")
    df = pd.read_csv(csv_path)
    return df

try:
    df_raw = load_data("loan_data.csv")
except Exception as e:
    st.error(f"Erreur lors du chargement du fichier : {e}")
    st.stop()

# --- Exploration initiale ---
st.sidebar.header("Contrôles")
show_data = st.sidebar.checkbox("Afficher le jeu de données (head)", value=False)
show_info = st.sidebar.checkbox("Afficher info & statistiques", value=False)

if show_data:
    st.subheader("Aperçu du jeu de données (head)")
    st.dataframe(df_raw.head())

if show_info:
    st.subheader("Informations & statistiques")
    st.write("Dimensions :", df_raw.shape)
    st.write(df_raw.describe(include="all").T)

st.markdown("---")

# --- Encodage des variables catégorielles ---
df = df_raw.copy()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# --- Définition de la cible ---
target_col = "loan_status"
if target_col not in df.columns:
    st.error(f"Colonne cible '{target_col}' introuvable dans le dataset.")
    st.stop()

X = df.drop(target_col, axis=1)
y = df[target_col]

# --- Options d'entraînement ---
st.sidebar.header("Entraînement & Modèles")
auto_train = st.sidebar.checkbox("Entraîner tous les modèles automatiquement", value=True)
test_size = st.sidebar.slider("Taille du test (%)", 10, 40, 20)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100.0, random_state=int(random_state)
)

train_btn = st.sidebar.button(" Entraîner maintenant") if not auto_train else False

# --- Entraînement des modèles ---
@st.cache_data
def train_models(X_train, X_test, y_train, y_test, X_full, y_full):
    # Régression Logistique
    log_model = LogisticRegression(max_iter=5000, solver="liblinear")
    log_model.fit(X_train, y_train)
    y_log_pred = log_model.predict(X_test)
    acc_log = accuracy_score(y_test, y_log_pred)
    report_log = classification_report(y_test, y_log_pred, output_dict=True)
    cm_log = confusion_matrix(y_test, y_log_pred)
    cv_log = cross_val_score(log_model, X_full, y_full, cv=5, scoring="accuracy")

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_rf_pred = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_rf_pred)
    report_rf = classification_report(y_test, y_rf_pred, output_dict=True)
    cm_rf = confusion_matrix(y_test, y_rf_pred)
    cv_rf = cross_val_score(rf_model, X_full, y_full, cv=5, scoring="accuracy")
    importances = rf_model.feature_importances_

    # SVM
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
        X_full_scaled, y_full, test_size=(X_test.shape[0]/X_full.shape[0]), random_state=42
    )
    svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=False)
    svm_model.fit(X_train_svm, y_train_svm)
    y_svm_pred = svm_model.predict(X_test_svm)
    acc_svm = accuracy_score(y_test_svm, y_svm_pred)
    report_svm = classification_report(y_test_svm, y_svm_pred, output_dict=True)
    cm_svm = confusion_matrix(y_test_svm, y_svm_pred)
    cv_svm = cross_val_score(svm_model, X_full_scaled, y_full, cv=5, scoring="accuracy")

    results = {
        "log": {"model": log_model, "acc": acc_log, "report": report_log, "cm": cm_log, "cv": cv_log},
        "rf": {"model": rf_model, "acc": acc_rf, "report": report_rf, "cm": cm_rf, "cv": cv_rf, "importances": importances},
        "svm": {"model": svm_model, "acc": acc_svm, "report": report_svm, "cm": cm_svm, "cv": cv_svm, "scaler": scaler},
    }
    return results

# --- Lancement de l'entraînement ---
if auto_train or train_btn:
    with st.spinner("🔧 Entraînement des modèles..."):
        results = train_models(X_train, X_test, y_train, y_test, X, y)
    st.success(" Entraînement terminé avec succès")
else:
    st.info("Cochez 'Entraîner automatiquement' ou cliquez sur 'Entraîner maintenant' pour lancer l'entraînement.")
    st.stop()

# --- Fonction d'affichage ---
def display_model_results(model_name, model_key, cmap):
    st.subheader(model_name)
    res = results[model_key]
    st.write(f"**Accuracy :** {res['acc']:.4f}")
    st.write("**Classification Report :**")
    st.dataframe(pd.DataFrame(res["report"]).T.round(3))

    st.write("**Matrice de confusion :**")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap=cmap, ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    st.pyplot(fig)

    st.write("**Validation croisée (5 folds)**")
    st.write(np.round(res["cv"], 3))
    st.write(f"Moyenne : {res['cv'].mean():.3f} ± {res['cv'].std():.3f}")

    if model_key == "rf":
        st.write("**Importance des variables :**")
        imp_df = pd.DataFrame({
            "feature": X.columns,
            "importance": res["importances"]
        }).sort_values(by="importance", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x="importance", y="feature", data=imp_df, palette="viridis", ax=ax)
        ax.set_title("Importance des variables - Random Forest")
        st.pyplot(fig)

# --- Onglets ---
st.header(" Analyse & Prédiction du défaut de prêt")
tabs = st.tabs([
    "Régression Logistique",
    "Random Forest",
    "SVM (RBF)",
    "Comparaison globale",
    " Prévision d’un emprunteur"
])

# --- Onglet 1 : Logistique ---
with tabs[0]:
    display_model_results("Régression Logistique", "log", "Blues")

# --- Onglet 2 : Random Forest ---
with tabs[1]:
    display_model_results("Random Forest", "rf", "Greens")

# --- Onglet 3 : SVM ---
with tabs[2]:
    display_model_results("SVM (RBF)", "svm", "Oranges")

# --- Onglet 4 : Comparaison ---
with tabs[3]:
    st.subheader("Comparaison des performances")
    df_results = pd.DataFrame({
        "Modèle": ["Régression Logistique", "Random Forest", "SVM (RBF)"],
        "Accuracy Test": [results["log"]["acc"], results["rf"]["acc"], results["svm"]["acc"]],
        "Accuracy Cross-Val": [results["log"]["cv"].mean(), results["rf"]["cv"].mean(), results["svm"]["cv"].mean()],
        "Écart-type CV": [results["log"]["cv"].std(), results["rf"]["cv"].std(), results["svm"]["cv"].std()]
    })
    st.dataframe(
        df_results.style.format({
            "Accuracy Test": "{:.4f}",
            "Accuracy Cross-Val": "{:.4f}",
            "Écart-type CV": "{:.4f}"
        })
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df_results, x="Accuracy Cross-Val", y="Modèle", palette="coolwarm", ax=ax)
    ax.set_title("Comparaison des modèles (Validation croisée)")
    st.pyplot(fig)

# --- Onglet 5 : Prévision interactive ---
with tabs[4]:
    st.subheader("Simulation de demandeur de prêt")

    input_data = {}
    for c in X.columns:
        if c in cat_cols:
            options = df_raw[c].astype(str).unique().tolist()
            choice = st.selectbox(f"{c}", options=options)
            input_data[c] = int(encoders[c].transform([choice])[0])
        else:
            median = float(df_raw[c].median())
            minv = float(df_raw[c].min())
            maxv = float(df_raw[c].max())
            step = (maxv - minv) / 100 if maxv > minv else 1.0
            input_data[c] = st.number_input(f"{c}", value=median, min_value=minv, max_value=maxv, step=step)

    model_choice = st.selectbox("Choisir le modèle", ["Random Forest", "Régression Logistique", "SVM (RBF)"])
    predict_btn = st.button(" Prédire le statut du prêt")

    if predict_btn:
        input_df = pd.DataFrame([input_data])[X.columns]

        if model_choice == "Random Forest":
            model = results["rf"]["model"]
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
        elif model_choice == "Régression Logistique":
            model = results["log"]["model"]
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
        else:
            model = results["svm"]["model"]
            scaler = results["svm"]["scaler"]
            input_scaled = scaler.transform(input_df.values)
            pred = model.predict(input_scaled)[0]
            proba = None

        st.markdown(f"###  Résultat : **{'Défaut de paiement' if pred == 1 else 'Aucun défaut'}**")
        if proba is not None:
            st.write(f"**Probabilités :** {np.round(proba, 3)}")
        else:
            st.info("Ce modèle ne fournit pas de probabilités (SVM sans probability=True).")

st.markdown("---")
st.info(" Cette application permet le nettoyage, l'entraînement, la comparaison et la prédiction du risque de défaut de prêt.")
