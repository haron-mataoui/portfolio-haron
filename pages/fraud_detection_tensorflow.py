# fraud_detection_tensorflow.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="💳 Détection de Fraude Bancaire - TensorFlow", 
    layout="wide"
)
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
    " Exploration des données",
    " Entraînement du modèle",
    " Évaluation et Visualisations",
    " Prédiction en direct"
])

# ----------------- TAB 1: EXPLORATION INTERACTIVE -----------------
with tabs[0]:
    st.header("Exploration interactive du Dataset")

    st.write("""
    Ce tableau présente un aperçu des premières lignes du dataset. 
    Chaque ligne correspond à une transaction bancaire et chaque colonne à une variable mesurée.
    """)
    st.dataframe(df.head(), use_container_width=True)

    st.write("Résumé statistique des colonnes numériques (moyenne, écart-type, min, max, etc.)")
    st.dataframe(df.describe().T, use_container_width=True)

    # Metrics clés
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de lignes", df.shape[0])
    col2.metric("Nombre de colonnes", df.shape[1])
    col3.metric("Transactions frauduleuses", int(df['Class'].sum()))

    st.write("### Distribution d’une colonne")
    col_to_plot = st.selectbox("Sélectionner une colonne à visualiser", df.columns)

    if pd.api.types.is_numeric_dtype(df[col_to_plot]):
        st.write(f"Histogramme de `{col_to_plot}`. L’échelle est logarithmique pour mieux visualiser les valeurs rares et extrêmes.")
        fig = px.histogram(df, x=col_to_plot, nbins=50, log_y=True, title=f"Histogramme de {col_to_plot}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(f"Distribution de `{col_to_plot}` (comptage des valeurs uniques).")
        fig = px.bar(df[col_to_plot].value_counts(), labels={'index': col_to_plot, 'value':'Count'}, title=f"Distribution de {col_to_plot}")
        st.plotly_chart(fig, use_container_width=True)

    st.write("""
    ### Carte de corrélation
    Cette heatmap montre les relations entre les différentes variables. 
    Les couleurs indiquent la force et le sens de la corrélation : 
    du bleu foncé (corrélation négative forte) au rouge foncé (corrélation positive forte).
    Les chiffres ont été retirés pour une lecture plus visuelle.
    """)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# ----------------- TAB 2: ENTRAÎNEMENT -----------------
with tabs[1]:
    st.header(" Entraînement du Modèle")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    st.write(f" Après SMOTE : {y_res.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    def build_model(input_dim):
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
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
    st.success(" Entraînement terminé !")

    st.subheader(" Architecture du modèle")
    st.write("""
    Ci-dessous, le résumé complet des couches du réseau de neurones. 
    Chaque ligne montre la couche, le type, sa fonction d'activation et le nombre de paramètres entraînables.
    Cette vue permet de comprendre la complexité et la structure du modèle.
    """)
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.text("\n".join(model_summary))

    # Sauvegarde dans session_state
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.history = history


# ----------------- TAB 3: ÉVALUATION -----------------
with tabs[2]:
    st.header("Évaluation du Modèle")

    if "model" in st.session_state:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        history = st.session_state.history

        # Courbes d’apprentissage
        st.write("""
        ### Courbes d’apprentissage
        Ces graphiques montrent l'évolution de la précision (accuracy) et de la perte (loss) 
        pendant l'entraînement et la validation. Ils permettent de détecter le surapprentissage 
        ou sous-apprentissage.
        """)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history['accuracy'], label='Train')
        ax[0].plot(history.history['val_accuracy'], label='Validation')
        ax[0].set_title('Précision (Accuracy)')
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend()

        ax[1].plot(history.history['loss'], label='Train')
        ax[1].plot(history.history['val_loss'], label='Validation')
        ax[1].set_title('Perte (Loss)')
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss")
        ax[1].legend()
        st.pyplot(fig)

        # Prédictions
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        acc = np.mean(y_pred.flatten() == y_test)
        auc = roc_auc_score(y_test, y_pred)
        st.write(f"**Accuracy sur le test set :** {acc:.4f} | **AUC :** {auc:.4f}")

        # Rapport de classification
        st.write("""
        ### Rapport de classification
        Le rapport détaille les métriques par classe (0 = normal, 1 = fraude) :
        précision, rappel (recall), f1-score et support (nombre d'exemples par classe).
        """)
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

        # Matrice de confusion
        st.write("""
        ### Matrice de confusion
        La matrice de confusion montre le nombre de prédictions correctes et incorrectes pour chaque classe. 
        Les vraies fraudes et faux positifs sont facilement identifiables.
        """)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title("Matrice de confusion")
        st.pyplot(fig)

        # Courbe ROC
        st.write("""
        ### Courbe ROC
        La courbe ROC (Receiver Operating Characteristic) permet de visualiser la performance 
        du modèle à différents seuils de classification. Plus la courbe est proche du coin supérieur gauche, 
        meilleur est le modèle.
        """)
        fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_title("Courbe ROC")
        ax.set_xlabel("Taux de faux positifs (FPR)")
        ax.set_ylabel("Taux de vrais positifs (TPR)")
        ax.legend()
        st.pyplot(fig)

    else:
        st.warning("⚠️ Entraîne d'abord le modèle dans l'onglet 'Entraînement'.")


# ----------------- TAB 4: PRÉDICTION -----------------
with tabs[3]:
    st.header(" Tester une Transaction")

    if "model" in st.session_state:
        model = st.session_state.model
        scaler = st.session_state.scaler

        input_data = {}
        for i, col in enumerate(df.columns[:-1]):  # exclude 'Class'
            input_data[col] = st.number_input(f"{col}", value=float(df[col].median()))

        if st.button(" Prédire fraude / normal"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            pred_proba = model.predict(input_scaled)[0][0]
            st.write(f"**Probabilité de fraude :** {pred_proba:.3f}")
            if pred_proba > 0.5:
                st.error(" Transaction potentiellement frauduleuse !")
            else:
                st.success(" Transaction probablement légitime.")
    else:
        st.warning("⚠️ Entraîne d'abord le modèle dans l'onglet 'Entraînement'.")
