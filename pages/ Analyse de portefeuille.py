import streamlit as st
from modules.analyse_exploratoire import afficher_analyse_exploratoire
from modules.simulation_portefeuille import afficher_simulation_portefeuille
from modules.prevision_actions import afficher_prevision_actions

st.title("Analyse de portefeuille")

# Création des sous-onglets
tab1, tab2, tab3 = st.tabs(["Analyse exploratoire", "Simulation", "Prévisions"])

with tab1:
    afficher_analyse_exploratoire()

with tab2:
    afficher_simulation_portefeuille()

with tab3:
    afficher_prevision_actions()
