# fraud_detection_tensorflow.py (Version avec graphiques Streamlit natifs)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Gardé UNIQUEMENT pour la matrice de confusion

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
# Cache la fonction pour ne charger les données qu'une seule fois
@st.cache_data
def load_data():
    """Charge les données depuis une URL et les retourne dans un DataFrame pandas."""
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# --- Définition des sous-onglets pour une navigation claire ---
tabs = st.tabs([
    " Exploration des données",
    " Entraînement du modèle",
    " Évaluation et Visualisations",
    " Prédiction en direct"
])

# ----------------- TAB 1: EXPLORATION -----------------
with tabs[0]:
    st.header("Exploration du Dataset")

    st.dataframe(df.head(), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de transactions", f"{df.shape[0]:,}")
    col2.metric("Transactions normales (Classe 0)", f"{df['Class'].value_counts()[0]:,}")
    col3.metric("Transactions frauduleuses (Classe 1)", f"{df['Class'].value_counts()[1]:,}")

    st.subheader("Distribution des montants des transactions")
    # Création d'un histogramme avec NumPy pour le passer à st.bar_chart
    # On se concentre sur les montants < 2500 pour une meilleure lisibilité
    hist_values = np.histogram(df['Amount'][df['Amount'] < 2500], bins=50)[0]
    st.bar_chart(hist_values, use_container_width=True)

    st.subheader("Distribution des classes (Normal vs. Fraude)")
    class_counts = df['Class'].value_counts()
    st.bar_chart(class_counts, use_container_width=True)


# ----------------- TAB 2: ENTRAÎNEMENT -----------------
with tabs[1]:
    st.header("Préparation des données et Entraînement du Modèle")

    if st.button("Lancer l'entraînement du modèle"):
        with st.spinner("Préparation des données en cours (Scaling, SMOTE)..."):
            X = df.drop("Class", axis=1)
            y = df["Class"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_scaled, y)
            st.success(f"Données rééquilibrées avec SMOTE : {y_res.value_counts().to_dict()}")

            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2, random_state=42
            )

        @st.cache_data # Cache le modèle pour éviter de le ré-entraîner
        def train_model(_X_train, _y_train):
            """Construit et entraîne le modèle TensorFlow."""
            model = models.Sequential([
                layers.Dense(64, activation='relu', input_shape=(_X_train.shape[1],)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
            
            history = model.fit(
                _X_train, _y_train,
                validation_split=0.2,
                epochs=20,
                batch_size=2048,
                verbose=0
            )
            return model, history

        with st.spinner(" Entraînement du réseau de neurones en cours..."):
            model, history = train_model(X_train, y_train)

        st.success(" Entraînement terminé !")

        st.subheader(" Architecture du modèle")
        model.summary(print_fn=lambda x: st.text(x))

        # Sauvegarde des objets nécessaires pour les autres onglets
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.history = history
        st.info("Le modèle est maintenant entraîné et prêt pour l'évaluation et la prédiction.")


# ----------------- TAB 3: ÉVALUATION -----------------
with tabs[2]:
    st.header("Évaluation du Modèle")

    if "model" in st.session_state:
        # Récupération des données depuis st.session_state
        model = st.session_state.model
        history = st.session_state.history
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        st.subheader("Courbes d'apprentissage")
        perf_df = pd.DataFrame(history.history)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Accuracy (Train vs Validation)")
            st.line_chart(perf_df[['accuracy', 'val_accuracy']])
        with col2:
            st.write("Loss (Train vs Validation)")
            st.line_chart(perf_df[['loss', 'val_loss']])

        # Prédictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        st.subheader("Métriques de performance")
        acc = np.mean(y_pred.flatten() == y_test.to_numpy())
        auc = roc_auc_score(y_test, y_pred_proba)
        
        c1, c2 = st.columns(2)
        c1.metric("Accuracy sur le test set", f"{acc:.4f}")
        c2.metric("AUC sur le test set", f"{auc:.4f}")
        
        st.subheader("Rapport de classification")
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
        st.dataframe(report_df, use_container_width=True)

        col_roc, col_cm = st.columns(2)
        with col_roc:
            st.subheader("Courbe ROC")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_df = pd.DataFrame({
                "Taux de faux positifs": fpr,
                "Taux de vrais positifs": tpr
            })
            st.area_chart(roc_df, x="Taux de faux positifs", y="Taux de vrais positifs", use_container_width=True)

        with col_cm:
            st.subheader("Matrice de confusion")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_xlabel("Prédit")
            ax.set_ylabel("Réel")
            st.pyplot(fig, use_container_width=True)
            
    else:
        st.warning(" Veuillez d'abord entraîner le modèle dans l'onglet ' Entraînement du modèle'.")

# ----------------- TAB 4: PRÉDICTION -----------------
with tabs[3]:
    st.header("Tester une Transaction sur une seule ligne")

    if "model" in st.session_state:
        # Récupération du modèle et du scaler
        model = st.session_state.model
        scaler = st.session_state.scaler

        st.write("Entrez les valeurs de la transaction à tester :")
        # Utilisation de st.form pour regrouper les inputs
        with st.form("prediction_form"):
            input_data = {}
            # Création de colonnes pour un affichage plus compact
            cols = st.columns(5)
            # Exclut 'Class' des colonnes à demander
            feature_columns = [col for col in df.columns if col != 'Class'] 
            for i, col_name in enumerate(feature_columns):
                input_data[col_name] = cols[i % 5].number_input(
                    f"{col_name}", value=float(df[col_name].median()), step=0.01
                )
            
            submitted = st.form_submit_button("🔍 Prédire fraude / normal")

        if submitted:
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            pred_proba = model.predict(input_scaled)[0][0]

            st.subheader("Résultat de la prédiction")
            if pred_proba > 0.5:
                st.error(f"🔴 Transaction potentiellement frauduleuse (Probabilité: {pred_proba:.2%})")
            else:
                st.success(f"🟢 Transaction probablement légitime (Probabilité de fraude: {pred_proba:.2%})")
    else:
        st.warning(" Veuillez d'abord entraîner le modèle dans l'onglet '🧠 Entraînement du modèle'.")
