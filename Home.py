import streamlit as st
import datetime

# --- Configuration de la page ---
st.set_page_config(page_title="Portfolio Haron Mataoui", page_icon="📊")

# --- Titre principal ---
st.title("🚀 Application Streamlit minimale")
st.write("Bravo ! Ton déploiement fonctionne correctement 🎉")

# --- Exemple d'interactivité ---
st.header("📅 Date et heure actuelles")
st.write(datetime.datetime.now())

# --- Exemple de widget ---
name = st.text_input("Quel est ton prénom ?")
if name:
    st.success(f"Salut {name}, ton app fonctionne parfaitement ✅")

# --- Graphique rapide ---
import pandas as pd
import numpy as np

data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C']
)
st.line_chart(data)
