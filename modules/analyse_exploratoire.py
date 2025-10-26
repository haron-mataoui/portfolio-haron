import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def afficher_analyse_exploratoire():
    st.subheader(" Analyse exploratoire des actions")

    # --- Sélection des actions et de la période ---
    tickers = st.multiselect(
        "Choisis des actions :", 
        ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA'], 
        default=['AAPL', 'MSFT', 'GOOG']
    )
    start = st.date_input("Date de début", pd.to_datetime("2020-01-01"))
    end = st.date_input("Date de fin", datetime.today())

    if not tickers:
        st.warning("Choisis au moins une action.")
        return

    # --- Téléchargement des données ---
    data = yf.download(tickers, start=start, end=end)["Close"]

    st.write("### Cours de clôture")
    st.line_chart(data)

    # --- Corrélation des rendements ---
    st.write("### Corrélation des rendements journaliers")
    returns = data.pct_change().dropna()
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    # --- Réduction de dimension : PCA ---
    st.subheader(" Analyse en Composantes Principales (PCA)")

    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns)

    n_components = st.slider("Nombre de composantes principales", 2, min(len(tickers), 5), 2)
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(returns_scaled)

    # --- Variance expliquée ---
    explained_variance = pca.explained_variance_ratio_
    st.write("**Variance expliquée par les composantes principales :**")
    st.bar_chart(pd.Series(explained_variance, index=[f"PC{i+1}" for i in range(n_components)]))

    # --- Visualisation sur les deux premières composantes ---
    if n_components >= 2:
        pca_df = pd.DataFrame(components[:, :2], columns=["PC1", "PC2"], index=returns.index)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x="PC1", y="PC2", data=pca_df, alpha=0.6)
        ax.set_title("Projection des rendements sur les deux premières composantes principales")
        st.pyplot(fig)

    # --- Contribution des actions ---
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=tickers
    )
    st.write("### Contribution (chargements) des actions aux composantes principales")
    st.dataframe(loadings.round(3))

    # Visualisation des contributions
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(loadings, annot=True, cmap="mako", ax=ax)
    ax.set_title("Contributions des actions aux composantes principales")
    st.pyplot(fig)
