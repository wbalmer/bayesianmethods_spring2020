from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
import seaborn as sb
sb.set()

# import dataset
data = pd.read_csv('TESS_planets.csv', skiprows=78)
physicals = ['pl_orbper', 'pl_orbsmax', 'pl_bmassj', 'pl_dens']
col_unit_dict = {"pl_orbper":"Orbital Period [days]",
    "pl_orbsmax":"Orbit Semi-Major Axis [au]",
    "pl_bmassj":"Planet Mass [Jupiter mass]",
    "pl_dens":"Planet Density [g/cm**3]"}

# ask for column assign data to X variable
col = input('Pick a dataset from the following : '+str(physicals)+'\n')
col = str(col)
X = data[col].dropna()
X = X.to_numpy().reshape(-1,1)

# Learn the best-fit GaussianMixture models
#  Here we'll use scikit-learn's GaussianMixture model. The fit() method
#  uses an Expectation-Maximization approach to find the best
#  mixture of Gaussians for the data

# fit models with 1-5 components
N = np.arange(1, 6)
models = [None for i in range(len(N))]

for i in range(len(N)):
    models[i] = GaussianMixture(N[i]).fit(X)

# compute the AIC and the BIC
AIC = [m.aic(X) for m in models]
BIC = [m.bic(X) for m in models]

list = []
mixn = input('enter your prefered number of gaussians (leave blank for minimization of BIC): ')
upper = input('enter the upper limit for the fit (leave blank for max of data): ')
if mixn == '':
    mixn = np.argmin(BIC)
if upper == '':
    upper = np.max(X)
list.append(int(mixn))
list.append(float(upper))
list

# limits and dists for plotting
def mixture_plot(mixn=np.argmin(BIC), upper=np.max(X)):
    lower = np.min(X)
    M_best = models[mixn]
    x = np.linspace(lower, upper, 5000)
    logprob = M_best.score_samples(x.reshape(-1, 1))
    responsibilities = M_best.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    fig = plt.figure(figsize=(7,5), dpi=150)
    # plot 1: data + best-fit mixture
    ax = plt.subplot2grid((6,2), (0,0), rowspan=4, colspan=2)

    ax.hist(X, 30, density=True, histtype='stepfilled',color='xkcd:mauve', alpha=0.5)
    ax.plot(x, pdf, '-',color='xkcd:light pink')
    ax.plot(x, pdf_individual, '-.', color='xkcd:rose', linewidth=1.5)
    ax.text(0.04, 0.96, "Best-fit Mixture",
        ha='left', va='top', transform=ax.transAxes)
    ax.set_xlabel(col_unit_dict[col])
    ax.set_ylabel('$p(x)$')
    ax.set_xlim(lower,upper)

    # plot 2: AIC and BIC
    ax = plt.subplot2grid((6,2), (4,0), rowspan=2)
    ax.plot(N, AIC, '-k', label='AIC')
    ax.plot(N, BIC, '--k', label='BIC')
    ax.set_xlabel('n. components')
    ax.set_ylabel('information criterion')
    ax.legend(loc=2)

    # plot 3: class probability
    ax = plt.subplot2grid((6,2), (4,1), rowspan=2)
    p = responsibilities
    p = p.cumsum(1).T
    for i in range(0,len(p)):
        if i == 0:
            ax.fill_between(x, 0, p[0], color='xkcd:mauve', alpha=0.2*(i+1))
        else:
            im = i-1
            ax.fill_between(x, p[im], p[i], color='xkcd:mauve', alpha=0.2*(i+1))
    ax.set_xlabel('Variable Sample')
    ax.set_ylabel('Probability from \ngiven gaussian')

    plt.tight_layout()
    plt.savefig('mixfit'+col+'.jpg', dpi=150)

mixture_plot(mixn=list[0],upper=list[1])
