# imports
import numpy as np # for computation and arrays
import pandas as pd # for datatables
import matplotlib.pyplot as plt # for plotting
import seaborn as sb # more plotting options
sb.set() #sets default plotting options, makes things look pretty
import statsmodels.api as sm # KDE package
from scipy import stats # more statistical functions
from scipy.stats import norm
from scipy.stats import shapiro

# read in data
data = pd.read_csv('Feb17th/TESS_planets.csv', skiprows=78)

# ask which column the user wants to plot
col = input('Pick a column. For options enter a question mark (?) : ')
col = str(col)
if col =='?':
    print(data.columns)
    col = input('Pick a column from the above list: ')

# ask which kernel to use
kern = str(input('Pick your kernel. Options are: gau, cos, tri, uni, biw, epa, triw. In doubt? try gau) : '))

# create the KDE for the column
d = data[col].dropna() # removes NaN values, which don't play nice
kde = sm.nonparametric.KDEUnivariate(d) # assign the statsmodels function to a smaller
if kern == 'gau':
    kde.fit(kernel=kern, bw='silverman') # fit the Kernel Estimate Densities
elif kern in ['cos', 'tri', 'uni', 'biw', 'epa', 'triw']:
    kde.fit(kernel=kern, fft=False, bw='silverman') # fit the Kernel Estimate Densities
else:
    print('That kernel isn\'t supported right now. Are you trying to cause problems?')

def plotthefig(save='n', g='n'):

    x = np.linspace(np.min(kde.support), np.max(kde.support), len(d)) # creates xvalues, range for plotting
    mu, std = norm.fit(d) # fits single peak gaussian to data

    weights = np.ones_like(d)/float(len(d))
    p = norm.pdf(x,mu,std)
    W, pW = shapiro(d)
    gaussianity = 'Data \n(Shapiro W='+str(round(W, 4))+')'
    # set plotting stuff
    f, ax = plt.subplots(figsize=(7,5),dpi=100)
    plt.xlabel(d.name)
    plt.ylabel('Occurrence (data) \nNormalized Likelihood (models)')
    plt.title('Kernel Density Estimation of column ' +d.name)

    ax.hist(d, bins=15, weights=weights, label=gaussianity, alpha=0.8, zorder=1, color='xkcd:mauve')
    sb.scatterplot(kde.support, kde.density/np.max(kde.density), zorder=2, color='xkcd:light pink', linewidth=0, s=20, label='KDE')
    if g == 'y':
        sb.lineplot(x,p/np.max(p),color='xkcd:dusty rose',linewidth=2, label='Normal fit (W=1)', zorder=1)
        h, l = ax.get_legend_handles_labels()
        plt.legend(handles=[h[1],h[0],h[2]], labels=[l[1],l[0],l[2]])
    else:
        h, l = ax.get_legend_handles_labels()
        plt.legend(handles=[h[1],h[0]], labels=[l[1],l[0]])

    if save == 'y':
        # defines a name for the figure to be generated
        img_name = 'TESS_'+d.name+'.jpg'
        # save figure to path
        plt.savefig(img_name, dpi=150)
        print('Your figure has been saved to: '+img_name)
    return

# ask if they'd like to save the plot
yn = str(input('Would you like to save the KDE fit (y/n)? '))
if yn == 'y':
    #ask if they'd like a gaussian fit for comparison
    gaus = str(input('Add Gaussian for comparison (y/n)? '))
    if gaus == 'y':
        plotthefig(save='y', g='y')
    else:
        plotthefig(save='y')
else:
    pass

# to quickly generate fits for all physical parameters, uncomment and run only this loop
#physicals = ['pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'pl_orbincl', 'pl_bmassj', 'pl_radj', 'pl_dens', 'st_dist', 'gaia_dist', 'st_optmag', 'st_teff', 'st_mass', 'st_rad']
#for col in physicals:
#    d = data[col].dropna()
#    kde = sm.nonparametric.KDEUnivariate(d)
#    kde.fit(kernel=kern, bw='silverman')
#    plotthefig(save='y', g='y')
