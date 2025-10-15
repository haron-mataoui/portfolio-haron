import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

def simulate_portfolio(initial_investment, mu, sigma, days):
    portfolio = np.zeros(days)
    portfolio[0] = initial_investment
    for i in range(1, days):
        daily_return = np.random.normal(mu, sigma)
        portfolio[i] = portfolio[i-1] * (1 + daily_return)
    return portfolio

def monte_carlo_simulation(initial_investment, mu, sigma, days, n_simulations):
    simulations = np.zeros((n_simulations, days))
    for i in range(n_simulations):
        simulations[i] = simulate_portfolio(initial_investment, mu, sigma, days)
    return simulations

def analyze_results(trajectories):
    final_values = trajectories[:, -1]
    mean_val = np.mean(final_values)
    median_val = np.median(final_values)
    VaR_5 = np.percentile(final_values, 5)
    return mean_val, median_val, VaR_5, final_values

st.title("Simulation Monte Carlo - Portefeuille Boursier")

st.markdown("""
### Qu'est-ce que la méthode de Monte Carlo ?

La **simulation Monte Carlo** est une technique statistique utilisée pour modéliser la probabilité d'événements futurs incertains.

Dans le cadre d'un portefeuille boursier, elle permet de simuler des milliers de trajectoires possibles de la valeur du portefeuille en fonction de distributions aléatoires des rendements journaliers.

### Comment ça marche ici ?

- Nous modélisons le rendement journalier comme une variable aléatoire suivant une loi normale, caractérisée par une moyenne (**mu**) et une volatilité (**sigma**) calculées à partir des données historiques du titre choisi.
- Pour chaque simulation, on génère un chemin possible de l'évolution du portefeuille sur un nombre de jours donné.
- En répétant cette simulation plusieurs fois (souvent des milliers), on obtient une distribution des valeurs finales possibles.
- À partir de cette distribution, on peut calculer des statistiques importantes comme la **valeur moyenne**, la **médiane**, ou encore la **Value at Risk (VaR)**, qui mesure le risque de perte avec un certain niveau de confiance.

Cette méthode est très utilisée en finance pour évaluer le risque et prévoir des scénarios futurs possibles.

---

*Après avoir lu cette introduction, vous pouvez configurer les paramètres dans la barre latérale et lancer la simulation.*
""")

# Sidebar inputs avec datetime.date
ticker = st.sidebar.text_input("Ticker boursier (ex: AAPL)", "AAPL")
start_date = st.sidebar.date_input("Date début", value=datetime.date(2024, 1, 1))
end_date = st.sidebar.date_input("Date fin", value=datetime.date(2025, 1, 1))
initial_investment = st.sidebar.number_input("Investissement initial (€)", min_value=100, value=10000)
days = st.sidebar.number_input("Nombre de jours de simulation", min_value=10, max_value=252, value=252)
n_simulations = st.sidebar.number_input("Nombre de simulations", min_value=100, max_value=10000, value=1000)

if st.sidebar.button("Lancer la simulation"):

    # Télécharger données
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("Aucune donnée disponible pour ce ticker / cette période.")
    else:
        data['Returns'] = data['Close'].pct_change().dropna()
        mu = data['Returns'].mean()
        sigma = data['Returns'].std()

        st.write(f"Rendement moyen journalier (mu): {mu:.6f}")
        st.write(f"Volatilité journalière (sigma): {sigma:.6f}")

        # Simulations Monte Carlo
        trajectories = monte_carlo_simulation(initial_investment, mu, sigma, days, n_simulations)
        mean_val, median_val, VaR_5, final_values = analyze_results(trajectories)

        # Affichage statistiques
        st.write(f"Valeur moyenne finale : {mean_val:.2f} €")
        st.write(f"Médiane finale : {median_val:.2f} €")
        st.write(f"Value at Risk (5%) : {VaR_5:.2f} €")

        # Graphique histogramme
        fig, ax = plt.subplots()
        ax.hist(final_values, bins=50, color='skyblue', edgecolor='black')
        ax.axvline(VaR_5, color='red', linestyle='--', label=f'VaR 5% = {VaR_5:.2f} €')
        ax.set_title("Distribution des valeurs finales du portefeuille")
        ax.set_xlabel("Valeur finale (€)")
        ax.set_ylabel("Fréquence")
        ax.legend()
        st.pyplot(fig)

        # Afficher quelques trajectoires
        fig2, ax2 = plt.subplots()
        for i in range(min(50, n_simulations)):
            ax2.plot(trajectories[i], alpha=0.3)
        ax2.set_title("Trajectoires simulées (50 premières)")
        ax2.set_xlabel("Jours")
        ax2.set_ylabel("Valeur (€)")
        st.pyplot(fig2)
