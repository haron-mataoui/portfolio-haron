import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc

from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="💳 Détection de Fraude Bancaire - TensorFlow",
    layout="wide"
)
st.title("💳 Détection de Fraude Bancaire avec TensorFlow ")


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
    Chaque ligne correspond à une transaction bancaire. La colonne `Class` indique si une transaction est frauduleuse (1) ou non (0).
    """)
    st.dataframe(df.head(), use_container_width=True)

    st.write("Résumé statistique des colonnes numériques.")
    st.dataframe(df.describe().T, use_container_width=True)

    # Metrics clés
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre total de transactions", f"{df.shape[0]:,}")
    col2.metric("Nombre de variables", df.shape[1])
    fraud_count = int(df['Class'].sum())
    fraud_percent = (fraud_count / df.shape[0]) * 100
    col3.metric("Transactions frauduleuses", f"{fraud_count} ({fraud_percent:.2f}%)")
    col4.metric("Transactions légitimes", f"{df.shape[0] - fraud_count:,}")


    st.write("### Distribution d’une colonne")
    # Définit l'index de la colonne 'Time' comme défaut
    try:
        default_ix = df.columns.get_loc('Time')
    except KeyError:
        default_ix = 0 # Fallback
    col_to_plot = st.selectbox("Sélectionner une colonne à visualiser", df.columns, index=default_ix)

    if pd.api.types.is_numeric_dtype(df[col_to_plot]):
        fig = px.histogram(df, x=col_to_plot, color="Class", nbins=50, log_y=True, title=f"Histogramme de {col_to_plot} par Classe")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Cette colonne n'est pas numérique.")

    st.write("""
    ### Carte de corrélation
    Cette heatmap montre les relations entre les différentes variables. La plupart des variables (`V1` à `V28`) sont le résultat d'une Analyse en Composantes Principales (ACP) et sont donc peu corrélées entre elles.
    """)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# ----------------- TAB 2: ENTRAÎNEMENT -----------------
with tabs[1]:
    st.header("Entraînement du Modèle")

    # --- Préparation des données ---
    st.subheader("1. Préparation et division des données")
    st.markdown("""
    La première étape consiste à séparer les features (`X`) de la cible (`y`).
    Ensuite, nous divisons les données en ensembles d'entraînement et de test.
    **C'est l'étape cruciale :** la division doit se faire *avant* toute technique de ré-échantillonnage pour éviter que le modèle ne "voie" des informations du jeu de test pendant son entraînement. Nous utilisons `stratify=y` pour conserver la même proportion de fraudes dans les deux ensembles.
    """)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Division AVANT le sur-échantillonnage pour éviter la fuite de données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    col1, col2 = st.columns(2)
    col1.metric("Taille du jeu d'entraînement", X_train.shape[0])
    col2.metric("Taille du jeu de test", X_test.shape[0])

    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # On applique la transformation apprise sur le train set

    # --- Gestion du déséquilibre ---
    st.subheader("2. Gestion du déséquilibre avec SMOTE")
    st.markdown("""
    Le jeu de données d'entraînement est **extrêmement déséquilibré** : il y a beaucoup plus de transactions légitimes que de fraudes. Si on entraînait le modèle directement dessus, il deviendrait "paresseux" et prédirait presque toujours "non-fraude", obtenant un score de précision élevé mais en étant inutile pour détecter les fraudes.

    Pour corriger cela, nous utilisons **SMOTE (Synthetic Minority Over-sampling Technique)**.
    """)
    with st.expander(" Cliquez ici pour comprendre comment fonctionne SMOTE en détail"):
        st.markdown("""
        Plutôt que de simplement dupliquer les rares exemples de fraude que nous avons, SMOTE est plus intelligent :
        
        1.  Il prend un exemple de fraude au hasard.
        2.  Il trouve ses voisins les plus proches dans l'espace des features (d'autres fraudes qui lui ressemblent).
        3.  Il choisit un de ces voisins et **crée un nouvel exemple synthétique** sur la ligne qui relie les deux.
        
        Imaginez que vous avez deux points bleus (fraudes) très proches sur un graphique. SMOTE va ajouter un nouveau point bleu quelque part sur le segment entre ces deux points. En répétant ce processus, on peuple l'ensemble d'entraînement avec des exemples de fraudes plausibles et variés, sans simplement copier les données existantes.

        **Résultat :** Le modèle dispose d'un jeu de données équilibré pour s'entraîner, ce qui l'oblige à apprendre les caractéristiques distinctives des fraudes de manière beaucoup plus efficace.
        
        **Important : SMOTE est appliqué UNIQUEMENT sur le jeu d'entraînement** pour ne pas introduire de données synthétiques dans notre ensemble de test, qui doit rester représentatif de la réalité.
        """)
    st.write(f"Distribution des classes **avant SMOTE** (dans le training set) :")
    st.json(pd.Series(y_train).value_counts().to_dict())

    if st.button("Lancer l'entraînement du modèle"):
        with st.spinner(" Application de SMOTE et entraînement du modèle en cours... Cela peut prendre quelques minutes."):
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            st.write(f"Distribution des classes **après SMOTE** (dans le training set) :")
            st.json(pd.Series(y_train_res).value_counts().to_dict())

            # --- Construction et entraînement du modèle ---
            st.subheader("3. Entraînement du réseau de neurones")

            def build_model(input_dim):
                model = models.Sequential([
                    layers.Input(shape=(input_dim,)),
                    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                    layers.Dropout(0.5),
                    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                    layers.Dropout(0.3),
                    layers.Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy',
                              metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
                return model

            model = build_model(X_train_res.shape[1])

            history = model.fit(
                X_train_res, y_train_res,
                validation_data=(X_test, y_test),
                epochs=20,
                batch_size=2048,
                verbose=0
            )
            st.success(" Entraînement terminé !")

            st.subheader("Architecture du modèle")
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text("\n".join(model_summary))

            # Sauvegarde dans session_state
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.history = history
            st.session_state.df_for_prediction = df.drop("Class", axis=1)


# ----------------- TAB 3: ÉVALUATION -----------------
with tabs[2]:
    st.header("Évaluation du Modèle sur le Jeu de Test Original")

    if "model" in st.session_state:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        history = st.session_state.history

        # Prédictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Rapport de classification
        st.subheader("Rapport de classification")
        
        st.info("""
        ### Comment interpréter ces résultats ?
        
        Ce tableau est le bulletin de notes le plus important pour notre modèle. L'accuracy seule est trompeuse ici. Voici comment lire les résultats pour la classe **1 (Fraude)** :

        -   **Recall (Rappel)** : **C'est la métrique la plus cruciale.** Un rappel élevé signifie que le modèle a réussi à **identifier un grand pourcentage des fraudes réelles**. Notre objectif principal est d'avoir ce chiffre le plus haut possible.

        -   **Precision (Précision)** : C'est le compromis. Une précision faible signifie que **lorsque le modèle sonne l'alarme, il peut souvent se tromper**. Le reste du temps, ce sont des "fausses alertes".

        -   **Le Dilemme :** C'est tout à fait normal ! Pour attraper un maximum de fraudes (haut rappel), le modèle doit être très sensible, ce qui génère mécaniquement plus de fausses alertes (basse précision). Les résultats que vous voyez sont donc **bons et réalistes** pour un système de détection de fraude.
        """)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).T.style.background_gradient(cmap='viridis', subset=['f1-score', 'precision', 'recall']))

        # Matrice de confusion
        st.subheader("Matrice de confusion")
        st.markdown("Visualise directement le nombre de bonnes et de mauvaises prédictions.")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=['Légitime', 'Fraude'], yticklabels=['Légitime', 'Fraude'])
        ax_cm.set_xlabel("Prédiction")
        ax_cm.set_ylabel("Valeur Réelle")
        ax_cm.set_title("Matrice de confusion")
        st.pyplot(fig_cm)

        # Courbes d’apprentissage
        st.subheader("Courbes d’apprentissage")
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[0].set_title('Précision (Accuracy)')
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend()

        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
        ax[1].set_title('Perte (Loss)')
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss")
        ax[1].legend()
        st.pyplot(fig)


        # Courbes ROC et Precision-Recall
        st.subheader("Courbes de performance")
        col1, col2 = st.columns(2)
        with col1:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], '--', color='gray')
            ax_roc.set_title("Courbe ROC")
            ax_roc.set_xlabel("Taux de faux positifs (FPR)")
            ax_roc.set_ylabel("Taux de vrais positifs (TPR)")
            ax_roc.legend()
            st.pyplot(fig_roc)

        with col2:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
            ax_pr.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
            ax_pr.set_title("Courbe Précision-Rappel")
            ax_pr.set_xlabel("Rappel (Recall)")
            ax_pr.set_ylabel("Précision (Precision)")
            ax_pr.legend()
            st.pyplot(fig_pr)


    else:
        st.warning(" Lancez d'abord l'entraînement du modèle dans l'onglet 'Entraînement du modèle'.")


# ----------------- TAB 4: PRÉDICTION -----------------
with tabs[3]:
    st.header("Tester une Transaction en Direct")


    st.info("""
    ###  Comprendre les variables `V1` à `V28`

    Les colonnes `V1` à `V28` ne représentent **pas des informations réelles** sur les transactions (comme le montant, le lieu, le type d’achat, etc.).  
    Elles proviennent d’une **Analyse en Composantes Principales (ACP / PCA)** réalisée pour **anonymiser les données d’origine** tout en conservant les structures statistiques.

    ####  Concrètement :
    - Chaque `Vn` est une **combinaison mathématique** de plusieurs caractéristiques d’origine (comme la fréquence d’achat, la catégorie du commerçant, l’heure, etc.).
    - Ces variables ont été **transformées pour la confidentialité** : leur sens exact n’est **pas interprétable directement**.
    - Par exemple, `V5` ne veut pas dire “montant” ni “type d’achat” : c’est une **dimension abstraite** du comportement transactionnel.

    ####  Pourquoi cela complique la prédiction manuelle :
    Dans une application réelle de détection de fraude, les variables `V1` à `V28` seraient **calculées automatiquement** à partir des données brutes d’une transaction via la même transformation PCA que celle utilisée pour entraîner le modèle.

    Ici, comme les variables originales sont inconnues, **l’utilisateur ne peut pas les saisir lui-même**.
    """)


    if "model" in st.session_state:
        model = st.session_state.model
        scaler = st.session_state.scaler
        df_cols = st.session_state.df_for_prediction

        st.write("Remplissez les champs ci-dessous pour simuler une nouvelle transaction. Les valeurs par défaut correspondent à la médiane de chaque variable.")

        input_data = {}
        # Créer une grille pour les inputs
        cols = st.columns(4)
        for i, col_name in enumerate(df_cols.columns):
            with cols[i % 4]:
                 input_data[col_name] = st.number_input(f"{col_name}", value=float(df_cols[col_name].median()), key=f"input_{col_name}")


        if st.button(" Analyser la transaction"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            pred_proba = model.predict(input_scaled)[0][0]

            st.subheader("Résultat de l'analyse")
            if pred_proba > 0.5:
                st.error(f"🔴 Transaction jugée **FRAUDULEUSE** avec une probabilité de {pred_proba:.2%}", icon="🚨")
            else:
                st.success(f"🟢 Transaction jugée **LÉGITIME** avec une probabilité de {1-pred_proba:.2%}", icon="✅")

            st.progress(float(pred_proba))
            st.write(f"Le score de probabilité de fraude est de **{pred_proba:.4f}**. Le seuil de classification est à 0.5.")
    else:
        st.warning(" Lancez d'abord l'entraînement du modèle dans l'onglet 'Entraînement du modèle'.")

