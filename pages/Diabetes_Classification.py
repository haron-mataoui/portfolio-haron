import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- CONFIGURATION ---
st.set_page_config(page_title="Analyse Diabète", page_icon="🩺", layout="wide")
st.title("🩺 Analyse du diabète – Comparaison complète de modèles ML")

# --- Chargement des données ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "diabetes.csv")
    df = pd.read_csv(csv_path)
    return df

df = load_data()

# --- Aperçu des données ---
st.subheader("Aperçu du jeu de données")
st.dataframe(df.head())

# --- Nettoyage des valeurs aberrantes ---
cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_invalid_zeros:
    df[col] = df[col].replace(0, df[col].median())

# --- Séparation des variables ---
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------------------------------
# --- Modèle 1 : Régression Logistique ---
# ------------------------------------------------------------------------------------------
st.header("📈 Modèle 1 : Régression Logistique")

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_log_pred = log_model.predict(X_test)

acc_log = accuracy_score(y_test, y_log_pred)
st.write(f"**Accuracy :** {acc_log:.3f}")

st.write("**Rapport de classification :**")
st.text(classification_report(y_test, y_log_pred))

# --- Matrice de confusion ---
fig1, ax1 = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_log_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title("Matrice de Confusion – Régression Logistique")
st.pyplot(fig1)

# --- Validation croisée ---
cv_log = cross_val_score(log_model, X, y, cv=5, scoring='accuracy')
st.write("**Validation croisée (5 folds)**")
st.write(f"Scores individuels : {np.round(cv_log, 3)}")
st.write(f"Score moyen : {cv_log.mean():.3f} ± {cv_log.std():.3f}")

st.write("---")

# ------------------------------------------------------------------------------------------
# --- Modèle 2 : Random Forest ---
# ------------------------------------------------------------------------------------------
st.header("🌲 Modèle 2 : Random Forest")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)

acc_rf = accuracy_score(y_test, y_rf_pred)
st.write(f"**Accuracy :** {acc_rf:.3f}")

st.write("**Rapport de classification :**")
st.text(classification_report(y_test, y_rf_pred))

# --- Matrice de confusion ---
fig2, ax2 = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_rf_pred), annot=True, fmt='d', cmap='Greens', ax=ax2)
ax2.set_title("Matrice de Confusion – Random Forest")
st.pyplot(fig2)

# --- Importance des variables ---
st.write("**Importance des variables :**")
importances = rf_model.feature_importances_
features = X.columns
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.barplot(x=importances, y=features, palette="viridis", ax=ax3)
ax3.set_title("Importance des Variables – Random Forest")
st.pyplot(fig3)

# --- Validation croisée ---
cv_rf = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
st.write("**Validation croisée (5 folds)**")
st.write(f"Scores individuels : {np.round(cv_rf, 3)}")
st.write(f"Score moyen : {cv_rf.mean():.3f} ± {cv_rf.std():.3f}")

st.write("---")

# ------------------------------------------------------------------------------------------
# --- Modèle 3 : SVM ---
# ------------------------------------------------------------------------------------------
st.header("🤖 Modèle 3 : SVM (Support Vector Machine)")

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train_svm, y_train_svm)
y_svm_pred = svm_model.predict(X_test_svm)

acc_svm = accuracy_score(y_test_svm, y_svm_pred)
st.write(f"**Accuracy :** {acc_svm:.3f}")

st.write("**Rapport de classification :**")
st.text(classification_report(y_test_svm, y_svm_pred))

# --- Matrice de confusion ---
fig4, ax4 = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test_svm, y_svm_pred), annot=True, fmt='d', cmap='Oranges', ax=ax4)
ax4.set_title("Matrice de Confusion – SVM")
st.pyplot(fig4)

# --- Validation croisée ---
cv_svm = cross_val_score(svm_model, X_scaled, y, cv=5, scoring='accuracy')
st.write("**Validation croisée (5 folds)**")
st.write(f"Scores individuels : {np.round(cv_svm, 3)}")
st.write(f"Score moyen : {cv_svm.mean():.3f} ± {cv_svm.std():.3f}")

st.write("---")

# ------------------------------------------------------------------------------------------
# --- Comparaison des modèles ---
# ------------------------------------------------------------------------------------------
st.header("📊 Comparaison des trois modèles")

df_results = pd.DataFrame({
    "Modèle": ["Régression Logistique", "Random Forest", "SVM (RBF)"],
    "Accuracy Test": [acc_log, acc_rf, acc_svm],
    "Accuracy Cross-Val": [cv_log.mean(), cv_rf.mean(), cv_svm.mean()],
    "Écart-type CV": [cv_log.std(), cv_rf.std(), cv_svm.std()]
})

st.dataframe(df_results.style.format({"Accuracy Test": "{:.3f}", "Accuracy Cross-Val": "{:.3f}", "Écart-type CV": "{:.3f}"}))

fig5, ax5 = plt.subplots(figsize=(8, 5))
sns.barplot(data=df_results, x="Accuracy Cross-Val", y="Modèle", palette="coolwarm", ax=ax5)
ax5.set_title("Comparaison des modèles (Validation croisée)")
st.pyplot(fig5)

st.success("✅ Analyse complète terminée ! Les trois modèles ont été comparés.")
