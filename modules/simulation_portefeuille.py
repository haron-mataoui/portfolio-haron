import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def afficher_simulation_portefeuille():
    st.subheader(" Simulation de portefeuille d’investissement")

    assets = st.multiselect("Actions :", ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA'], default=['AAPL', 'MSFT'])
    capital = st.number_input("Montant total à investir ($)", min_value=1000, step=1000, value=10000)
    start_date = st.date_input("Début", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("Fin", datetime.today())

    if not assets:
        st.warning("Sélectionne au moins une action.")
        return

    data = yf.download(assets, start=start_date, end=end_date)["Close"]
    normalized = data / data.iloc[0]
    portfolio = normalized.mean(axis=1) * capital

    st.line_chart(portfolio)
    st.markdown(f"**Valeur finale :** ${portfolio.iloc[-1]:,.2f}")
