# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm
from scipy import stats

# read in data
data = pd.read_csv('TESS_planets.csv', skiprows=78)
data.columns
sb.set()
x = data['st_mass'].dropna()
kde = sm.nonparametric.KDEUnivariate(x)
kde.fit(kernel='uni', fft=False, bw='silverman') # Estimate the densities
true = stats.norm.pdf(x=x)

fig, ax = plt.subplots(figsize=(7,5))
sb.scatterplot(x, true, label='TESS planets')
plt.xlim(np.min(x), np.max(x))
plt.legend()
plt.xlabel('Host mass')
plt.ylabel('Occurrence')
plt.title('Kernel Density Estimation of TESS discovered planet host masses')
plt.savefig('TESS_hostmasses.jpg', dpi=100)
