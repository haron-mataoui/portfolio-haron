import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import matplotlib.pyplot as plt

def afficher_prevision_actions():
    st.subheader("Prévision linéaire des prix d’actions")

    ticker = st.selectbox(
        "Choisis une action :",
        ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA'],
        key="ticker_prevision"  # clé unique
    )

    start_date = st.date_input(
        "Début",
        pd.to_datetime("2020-01-01"),
        key="date_debut_prevision"  # clé unique
    )

    end_date = st.date_input(
        "Fin",
        datetime.today(),
        key="date_fin_prevision"  # clé unique
    )

    horizon = st.slider(
        "Jours à prédire",
        10, 100, 30,
        key="slider_prevision"  # clé unique
    )

    # --- Téléchargement et modèle
    df = yf.download(ticker, start=start_date, end=end_date)[["Close"]]
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values
    model = LinearRegression().fit(X, y)

    X_future = np.arange(len(df), len(df)+horizon).reshape(-1, 1)
    y_future = model.predict(X_future)
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon)

    # --- Graphique
    fig, ax = plt.subplots()
    ax.plot(df.index, y, label="Historique")
    ax.plot(future_dates, y_future, label="Prévision", color="red")
    ax.legend()
    st.pyplot(fig)
