# science imports
import numpy as np
import pandas as pd
import scipy.stats as stat
from pymc3 import  *
import linmix

# plotting imports
import matplotlib.pyplot as plt
import matplotlib.markers
import seaborn as sb
import corner

# set up data
def line(x, slope, intercept):
    y = slope*x + intercept
    return y

data = pd.read_csv('TESS_recoveries_data.csv', index_col='Number')
data2 = data.drop(columns=['RA', 'Dec', 'Type', 'Ha recovery', 'Spectral Type', 'Gaia B-R', '( Don\'t use)TESS satellite period (days)', 'Ha period', 'Ha-off period'])
data3 = data2.fillna(np.nanmedian(data['Cont amp err']))

data4 = data.dropna()
x = data4['Ha Amplitude'].to_numpy()
y = data4['Cont amplitude'].to_numpy()

lims = []
lims_e = []
for targ in data2.index:
    if np.isnan(data2['Cont amplitude'][targ]):
        lims.append((data2['Ha Amplitude'][targ]))
        lims_e.append(np.nanmedian(data['Cont amp err']))

# stats regression
slope, intercept1, r_value, p_value, std_err = stat.linregress(x, y)
regression_line = line(x, slope, intercept1)

print('Beginning basic linear regression MCMC')
# basic linear regression MCMC
with Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    sigma = HalfNormal('sigma')#, beta=10, testval=std_err)
    intercept = Normal('Intercept', 0, sigma=20)
    x_coeff = Normal('x', 0, sigma=20)

    # Define likelihood
    likelihood = Normal('y', mu=intercept + x_coeff * x,
                        sigma=sigma, observed=y)

    # Inference!
    trace = sample(300, chains=1, cores=2) # draw 3000 posterior samples using NUTS sampling

plt.figure(figsize=(7, 7))
traceplot(trace[0:])
plt.tight_layout()
plt.title('Linear Regression MCMC Results')
plt.savefig('var_mcmc_regression_results.png')

plt.figure(figsize=(7, 7))
plt.plot(x, y, 'x', label='data')
plot_posterior_predictive_glm(trace, samples=300,
                              label='posterior predictive regression lines')
plt.plot(x, regression_line, label='scipy regression line', lw=3., c='r')

plt.title('Posterior predictive regression lines')
plt.legend(loc=0)
plt.xlim(0.05,0.4)
plt.ylim(0,0.4)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('var_mcmc_regression.png')


# run right censored regression MCMC fit
print('Now running LinMix right censored regression MCMC: \n')
lm = linmix.LinMix(data3['Cont amplitude'], data3['Ha Amplitude'], xsig=data3['Cont amp err'], ysig=data3['Ha amp err'], delta=data3['Ha-off recovery'], seed=np.random.randint(300), parallelize=False)
lm.run_mcmc(miniter=1000, maxiter=5000)

temp_arr = np.array([lm.chain[:]['alpha'], lm.chain[:]['beta'], lm.chain[:]['sigsqr'], lm.chain[:]['corr']])
temp_arr2 = temp_arr.reshape(temp_arr.shape[1], temp_arr.shape[0])

print('Now generating figures')
corn_fig = corner.corner(temp_arr2, labels=['$\\alpha$', '$\\beta$', '$\\sigma^2$', 'correlation'], range=[(0.1,3),(0.1,4),(0.1,4),(0.1,3)])
corn_fig.suptitle('LinMix Corner Plot', size=16)
corn_fig.savefig('mcmc_censored_corner.png')


# plotting results

slope_max = np.mean(lm.chain[:]['beta']) + 5*np.mean(lm.chain[:]['sigsqr'])
slope_min = np.mean(lm.chain[:]['beta']) - 5*np.mean(lm.chain[:]['sigsqr'])

plt.figure(figsize=(7,7), dpi=150)

x = np.arange(0, 0.8, 0.01)
x1 = np.zeros(len(lims))

sb.scatterplot(data2['Cont amplitude'], data2['Ha Amplitude'], color='xkcd:rose', label='Recovered in both filters')
plt.scatter(lims_e, lims, marker=matplotlib.markers.CARETLEFT, color = 'xkcd:rose', label='Recovered in H$\\alpha$ (continuum non-det.)')

plt.plot(x, line(x, 1, 0), 'k--', alpha=0.5)

for i in range(0, len(lm.chain), 5):
    xs = np.arange(0,.8,0.1)
    ys = lm.chain[i]['alpha'] + xs * lm.chain[i]['beta']
    plt.plot(xs, ys, color='xkcd:light pink', alpha=0.01)

mcmc_label = 'MCMC fit slope: '+str(round(np.mean(lm.chain[:]['beta']), 3))+' +/- '+str(round(5*np.mean(lm.chain[:]['sigsqr']), 3))

plt.plot(x, line(x, np.mean(lm.chain[:]['beta']), np.mean(lm.chain[:]['alpha'])), label=mcmc_label, color='xkcd:mauve')
plt.fill_between(x, line(x, slope_min, np.mean(lm.chain[:]['alpha'])), line(x, slope_max, np.mean(lm.chain[:]['alpha'])), alpha=0.7, color='xkcd:mauve')

plt.xlim(0.03, 0.7)
plt.ylim(0.03, 0.7)
plt.xlabel('H$\\alpha$-off amplitude (Norm. relative flux)')
plt.ylabel('H$\\alpha$ amplitude (Norm. relative flux)')
plt.legend()
plt.title('TESS H$\\alpha$ variation vs H$\\alpha$-off variation')

plt.savefig('final_mcmc_regression.png', dpi=500)
