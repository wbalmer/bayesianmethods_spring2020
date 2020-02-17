# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()
import statsmodels.api as sm
from scipy import stats

# read in data
data = pd.read_csv('TESS_planets.csv', skiprows=78)

# ask which column the user wants to plot
col = input('Pick a column. For options enter a question mark (?) : ')
col = str(col)
if col =='?':
    print(data.columns)
    col = input('Pick a column from the above list: ')

# ask which kernel to use
kern = str(input('Pick your kernel. Options are: gau, cos, tri, uni, biw, epa, triw. In doubt? try gau)'))

# create the KDE for the column
x = data[col].dropna() # removes NaN values, which don't play nice
kde = sm.nonparametric.KDEUnivariate(x) # assign the statsmodels function to a smaller
if kern == 'gau':
    kde.fit(kernel=kern, bw='silverman') # fit the Kernel Estimate Densities
elif kern in ['cos', 'tri', 'uni', 'biw', 'epa', 'triw']:
    kde.fit(kernel=kern, fft=False, bw='silverman') # fit the Kernel Estimate Densities
else:
    print('That kernel isn\'t supported right now. Are you trying to cause problems?')
pdf = stats.norm.pdf(x=x) # creates pdf values for a single normal model

def plotthefig(save='n', g='n'):
    fig, ax = plt.subplots(figsize=(7,5))
    # plots KDE
    sb.scatterplot(kde.support, kde.density/np.max(kde.density), label='KDE')
    # plots simple normal fit if kwarg is 'y'
    if g == 'y':
        sb.scatterplot(x, pdf, alpha=0.5, label='gaussian fit')
    # set plotting stuff
    plt.xlim(np.min(x), np.max(x))
    plt.legend()
    plt.xlabel(x.name)
    plt.ylabel('Occurrence')
    plt.title('Kernel Density Estimation of column ' +x.name)
    if save == 'y':
        # defines a name for the figure to be generated
        img_name = 'TESS_'+x.name+'.jpg'
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
