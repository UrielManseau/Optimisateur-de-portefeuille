from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

actifs = []
nombre_titres=int(input("Entrez le nombre de titres dans le portefeuille : "))
print("Entrez maintenant les symboles des titres dans le portfeuille : ")
for i in range(nombre_titres):
    symbole=input()
    actifs.append(symbole)

poids_initial = 1/nombre_titres
poids = np.full(nombre_titres,poids_initial)

fin = datetime.today().strftime("%Y-%m-%d")
debut_portefeuille = datetime(int(input("Année du plus récent IPO : ")), int(input("Month of the most recent IPO : ")), int(input("Day of the most recent IPO :")))
df = pd.DataFrame()
for stock in actifs :
    df[stock] = web.DataReader(stock, data_source = 'yahoo', start = debut_portefeuille, end=fin)['Adj Close']

titre = 'Historique Prix fermeture Portefeuille'
mesTitres = df
for colonnes in mesTitres.columns.values:
    plt.plot(mesTitres[colonnes], label = colonnes)

plt.title(titre)
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Prix fermeture (USD $)', fontsize =18)
plt.legend(mesTitres.columns.values, loc = 'upper left')
plt.show()

#Rendement journalier
rendement = df.pct_change()
rendement

#Matrice des covariances des rendements (252 = nb de jours de trading pour un an)
covMatriceAnnuel = rendement.cov() * 252
covMatriceAnnuel

#Calcul variance, volatilité et rendement annuel portefeuille
variance_portefeuille = np.dot(poids.T, np.dot(covMatriceAnnuel, poids))
variance_portefeuille
volatility_portefeuille = np.sqrt(variance_portefeuille)
rendement_annuel_portefeuille = np.sum(rendement.mean() * poids * 252)

#Montrer le rendement annuel attendu, la volatilité (le risque) et la variance
pourcentage_variance = str(round(variance_portefeuille, 2)*100) + "%"
pourcentage_vola = str(round(volatility_portefeuille, 2)*100) + "%"
pourcentage_rendement = str(round(rendement_annuel_portefeuille, 2)*100) + "%"
print('')
print ('Voici la performance et volatilité historique du portefeuille avec des poids équivalents : ')
print('Rendement historique: ' + pourcentage_rendement)
print('Volatilité historique annuelle: ' + pourcentage_vola)
print('Variance historique annuelle: ' + pourcentage_variance)

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

#Optimiser pour maximiser le ratio sharpe (mesure la performance d'un investissement avec risque vs un investissement sans risque comme obligation)
ef = EfficientFrontier(mu,S)
poids = ef.max_sharpe()
adjusted_weights = ef.clean_weights()
print('')
print('Poids optimaux : ', adjusted_weights)
print('')
print('Voici maintenant la perfomance historique avec une répartition des actifs maximisant le ratio sharpe : ')
ef.portfolio_performance(verbose = True)

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
derniers_prix = get_latest_prices(df)
poids = adjusted_weights
Valeur_portefeuille = int(input("Entrez la valeur du portefeuille en USD$ : "))
da = DiscreteAllocation(poids, derniers_prix, Valeur_portefeuille)
allocation, reste = da.lp_portfolio()
print('Allocation:', allocation)
print('Fonds restants: ${:.2f}'.format(reste))









