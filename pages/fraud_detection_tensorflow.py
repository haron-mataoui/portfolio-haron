# fraud_detection_tensorflow.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------- CONFIG -----------------
st.set_page_config(page_title="💳 Détection de Fraude Bancaire - TensorFlow", layout="wide")
st.title("💳 Détection de Fraude Bancaire avec TensorFlow")

# ----------------- DATA LOADING -----------------
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# --- Définition des sous-onglets ---
tabs = st.tabs([
    "📊 Exploration des données",
    "🧠 Entraînement du modèle",
    "📈 Évaluation et Visualisations",
    "🧾 Prédiction en direct"
])

# ----------------- TAB 1: EXPLORATION -----------------
with tabs[0]:
    st.header("📊 Exploration du Dataset")

    st.write("### Aperçu des données :")
    st.dataframe(df.head())

    st.write(f"**Taille du dataset :** {df.shape}")
    st.write("**Colonnes :**", list(df.columns))

    # Graphiques
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df['Amount'], bins=40, ax=ax[0])
    ax[0].set_title("Distribution des montants")
    sns.countplot(x='Class', data=df, ax=ax[1])
    ax[1].set_title("Distribution des classes (0 = normal, 1 = fraude)")
    st.pyplot(fig)

# ----------------- TAB 2: ENTRAÎNEMENT -----------------
with tabs[1]:
    st.header("🧠 Entraînement du Modèle")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    st.write(f"✅ Après SMOTE : {y_res.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    def build_model(input_dim):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC()])
        return model

    model = build_model(X_train.shape[1])

    with st.spinner("⏳ Entraînement du modèle en cours..."):
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=20,
            batch_size=2048,
            verbose=0
        )
    st.success("✅ Entraînement terminé !")

    st.subheader("📚 Architecture du modèle")
    model.summary(print_fn=lambda x: st.text(x))

    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.history = history

# ----------------- TAB 3: ÉVALUATION -----------------
with tabs[2]:
    st.header("📈 Évaluation du Modèle")

    if "model" in st.session_state:
        model = st.session_state.model
        scaler = st.session_state.scaler
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        history = st.session_state.history

        # Courbes d’apprentissage
        st.write("### Courbes d’apprentissage")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history['accuracy'], label='Train')
        ax[0].plot(history.history['val_accuracy'], label='Validation')
        ax[0].set_title('Accuracy')
        ax[0].legend()

        ax[1].plot(history.history['loss'], label='Train')
        ax[1].plot(history.history['val_loss'], label='Validation')
        ax[1].set_title('Loss')
        ax[1].legend()
        st.pyplot(fig)

        # Prédictions
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        acc = np.mean(y_pred.flatten() == y_test)
        auc = roc_auc_score(y_test, y_pred)
        st.write(f"**Accuracy :** {acc:.4f} | **AUC :** {auc:.4f}")

        st.write("### Rapport de classification")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title("Matrice de confusion")
        st.pyplot(fig)

        # Courbe ROC
        fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_title("Courbe ROC")
        ax.set_xlabel("Faux positifs")
        ax.set_ylabel("Vrais positifs")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("⚠️ Entraîne d'abord le modèle dans l'onglet précédent.")

# ----------------- TAB 4: PRÉDICTION -----------------
with tabs[3]:
    st.header("🧾 Tester une Transaction")

    if "model" in st.session_state:
        model = st.session_state.model
        scaler = st.session_state.scaler

        input_data = {}
        for i, col in enumerate(df.columns[:-1]):  # exclude 'Class'
            input_data[col] = st.number_input(f"{col}", value=float(df[col].median()))

        if st.button("🔍 Prédire fraude / normal"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            pred_proba = model.predict(input_scaled)[0][0]
            st.write(f"**Probabilité de fraude :** {pred_proba:.3f}")
            if pred_proba > 0.5:
                st.error("⚠️ Transaction potentiellement frauduleuse !")
            else:
                st.success("✅ Transaction probablement légitime.")
    else:
        st.warning("⚠️ Entraîne d'abord le modèle dans l'onglet '🧠 Entraînement'.")
