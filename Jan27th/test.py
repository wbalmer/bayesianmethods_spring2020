# This script is based off this seaborn example: https://seaborn.pydata.org/examples/cubehelix_palette.html
# William Balmer, 2/7/2020

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="dark")
rs = np.random.RandomState(666)

# Set up the matplotlib figure
f, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

# Rotate the starting point around the cubehelix hue circle
for ax, s in zip(axes.flat, np.linspace(0, 3, 10)):

    # Create a cubehelix colormap to use with kdeplot
    cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)

    # Generate and plot a random bivariate dataset
    x, y = rs.randn(2, 50)
    sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=ax)
    ax.set(xlim=(-3, 3), ylim=(-3, 3))

f.tight_layout()
f.savefig('test.png', dpi=200)
print('saved test kdeplots. have a nice day :)')
