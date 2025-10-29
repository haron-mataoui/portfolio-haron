# pages/Diabetes_Classification.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from streamlit_option_menu import option_menu

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuration ---
st.set_page_config(page_title="Diabetes - Analysis & Demo", page_icon="🩺", layout="wide")
st.title("🩺 Prediction of diabetes — Comprehensive analysis & interactive demonstration")

# --- Chargement robuste du CSV ---
@st.cache_data
def load_data(csv_name="diabetes.csv"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier '{csv_name}' est introuvable dans {current_dir}")
    return pd.read_csv(csv_path)

try:
    df = load_data("diabetes.csv")
except Exception as e:
    st.error(f"Erreur lors du chargement du fichier : {e}")
    st.stop()

# --- Nettoyage des valeurs aberrantes ---
cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_invalid_zeros:
    df[col] = df[col].replace(0, df[col].median())

# --- Séparation X / y ---
target_col = "Outcome"
if target_col not in df.columns:
    st.error(f"Colonne cible '{target_col}' introuvable.")
    st.stop()

X = df.drop(target_col, axis=1)
y = df[target_col]

# --- Sidebar ---
with st.sidebar:
    st.header(" Training settings")
    show_data = st.checkbox(" Display the dataset", value=False)
    auto_train = st.checkbox(" Train automatically", value=True)
    test_size = st.slider("Test size (%)", 10, 40, 20)
    random_state = st.number_input("Random state", value=42, step=1)
    train_btn = st.button(" Start training") if not auto_train else False

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100.0, random_state=int(random_state)
)

# --- Fonction d'entraînement ---
@st.cache_data
def train_models(X_train, X_test, y_train, y_test, X_full, y_full):
    # Régression logistique
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
    svm_model = SVC(kernel='rbf', C=1, gamma='scale')
    svm_model.fit(X_train_svm, y_train_svm)
    y_svm_pred = svm_model.predict(X_test_svm)
    acc_svm = accuracy_score(y_test_svm, y_svm_pred)
    report_svm = classification_report(y_test_svm, y_svm_pred, output_dict=True)
    cm_svm = confusion_matrix(y_test_svm, y_svm_pred)
    cv_svm = cross_val_score(svm_model, X_full_scaled, y_full, cv=5, scoring="accuracy")

    return {
        "log": {"model": log_model, "acc": acc_log, "report": report_log, "cm": cm_log, "cv": cv_log},
        "rf": {"model": rf_model, "acc": acc_rf, "report": report_rf, "cm": cm_rf, "cv": cv_rf, "importances": importances},
        "svm": {"model": svm_model, "acc": acc_svm, "report": report_svm, "cm": cm_svm, "cv": cv_svm, "scaler": scaler}
    }

# --- Entraînement ---
if auto_train or train_btn:
    with st.spinner(" Training of models in progress..."):
        results = train_models(X_train, X_test, y_train, y_test, X, y)
    st.success(" Training completed successfully !")
else:
    st.info("Tick 'Train automatically' or click 'Start training'.")
    st.stop()

# --- Fonction d'affichage des résultats ---
def display_model_results(model_name, model_key, cmap):
    st.subheader(model_name)
    res = results[model_key]
    st.write(f"** Accuracy :** {res['acc']:.4f}")
    st.dataframe(pd.DataFrame(res["report"]).T.round(3))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap=cmap, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Real")
    st.pyplot(fig)
    st.write(f"** Cross-validation (5 folds)** : {res['cv'].mean():.3f} ± {res['cv'].std():.3f}")

    if model_key == "rf":
        st.write(" **Importance of variables :**")
        imp_df = pd.DataFrame({
            "feature": X.columns,
            "importance": res["importances"]
        }).sort_values(by="importance", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x="importance", y="feature", data=imp_df, palette="viridis", ax=ax)
        ax.set_title("Importance of variables - Random Forest")
        st.pyplot(fig)

# --- MENU PRINCIPAL AVEC ÉMOJIS ---
selected = option_menu(
    menu_title="🩺 Navigation",
    options=[
        " Logistic Regression",
        " Random Forest",
        " SVM (RBF)",
        " Overall comparison",
        " Patient forecast"
    ],
    icons=["bar-chart", "tree", "activity", "columns-gap", "person"],
    menu_icon="stethoscope",
    default_index=0,
    orientation="horizontal",
)

# --- Navigation via le menu ---
if selected == " Logistic Regression":
    display_model_results(" Logistic Regression", "log", "Blues")

elif selected == " Random Forest":
    display_model_results(" Random Forest", "rf", "Greens")

elif selected == " SVM (RBF)":
    display_model_results(" SVM (RBF)", "svm", "Oranges")

elif selected == " Overall comparison":
    st.subheader(" Model comparison")
    df_results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "SVM (RBF)"],
        "Accuracy Test": [results["log"]["acc"], results["rf"]["acc"], results["svm"]["acc"]],
        "Accuracy CV (moy.)": [results["log"]["cv"].mean(), results["rf"]["cv"].mean(), results["svm"]["cv"].mean()],
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
    sns.barplot(data=df_results, x="Accuracy CV (moy.)", y="Model", palette="coolwarm", ax=ax)
    ax.set_title("Model comparison (Cross-validation)")
    st.pyplot(fig)

elif selected == " Patient forecast":
    st.subheader(" Enter the patient's characteristics")
    input_data = {}
    for c in X.columns:
        median, minv, maxv = float(df[c].median()), float(df[c].min()), float(df[c].max())
        step = (maxv - minv) / 100 if maxv > minv else 1.0
        input_data[c] = st.number_input(f"{c}", value=median, min_value=minv, max_value=maxv, step=step)

    pred_model_choice = st.selectbox("Select the model", [" Random Forest", " Logistic Regression", " SVM (RBF)"])
    predict_btn = st.button(" Predicting diabetes")

    if predict_btn:
        input_df = pd.DataFrame([input_data])[X.columns]
        if pred_model_choice == " Random Forest":
            model = results["rf"]["model"]
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
        elif pred_model_choice == " Logistic Regression":
            model = results["log"]["model"]
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
        else:
            model = results["svm"]["model"]
            scaler = results["svm"]["scaler"]
            input_scaled = scaler.transform(input_df.values)
            pred = model.predict(input_scaled)[0]
            proba = None

        st.markdown(f"### 🩺 Result : {'**🟥 Diabetic**' if pred == 1 else '**🟩 Non-diabetic**'}")
        if proba is not None:
            st.write(f"**Probabilities :** {np.round(proba, 3)}")
        else:
            st.info("This model does not provide probabilities. (SVM sans probability=True).")

st.markdown("---")
st.caption(" Diabetes analysis application: cleaning, training, comparison and prediction.")
