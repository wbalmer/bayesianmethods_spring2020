import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import norm
from scipy.stats import shapiro
import statsmodels.api as sm

data = pd.read_csv('Feb17th/TESS_planets.csv', skiprows=78)
d2 = data['pl_bmassj'].dropna()
kde = sm.nonparametric.KDEUnivariate(d2)
kde.fit(kernel='gau', bw='silverman')
mu, std = norm.fit(d2)
x = np.linspace(np.min(kde.support), np.max(kde.support), len(d2))
weights = np.ones_like(d2)/float(len(d2))
p = norm.pdf(x,mu,std)
W, pW = shapiro(d2)
gaussianity = 'Data \n(Shapiro W='+str(round(W, 4))+')'

f, ax = plt.subplots(figsize=(7,5),dpi=100)
ax.hist(d2, bins=15, weights=weights, label=gaussianity, alpha=0.8, zorder=1)
sb.scatterplot(kde.support, kde.density/np.max(kde.density), zorder=2, color='xkcd:mustard', linewidth=0.3, label='KDE')
sb.lineplot(x,p,color='xkcd:rose',linewidth=2, label='Normal fit (W=1)', zorder=1)
h, l = ax.get_legend_handles_labels()
plt.legend(handles=[h[1],h[0],h[2]], labels=[l[1],l[0],l[2]])
