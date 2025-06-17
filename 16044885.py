# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:50:26 2024

@author: brend

Small project for FDS
"""

import numpy as np;
import matplotlib.pyplot as plt;

data2020 = np.genfromtxt('2020input5.csv');

# Separating into 2 arrays, bins and values.
bins2020 = np.concatenate((data2020[[0],[0]], data2020[:,1]));
bVal2020 = data2020[:,2];

data2024 = np.genfromtxt('2024input5.csv');
counts2024 = np.unique(data2024, return_counts=True);

# Calculation of means, standard deviations and V-value.
mean20 = np.sum((bins2020[:-1] + 2) * bVal2020) / np.sum(bVal2020);
mean24 = np.sum(data2024) / len(data2024);
var20 = np.sum((bins2020[:-1] + 2 - mean20)**2 * bVal2020) / np.sum(bVal2020);
sd20 = np.sqrt(var20);
var24 = np.sum((data2024 - mean24)**2) / len(data2024);
sd24 = np.sqrt(var24);

# V-value = proportion scoring grade of 50 or higher in the 2020 exam.
# Absolute number (owing to the resolution of bin width an approximation):
plus50 = (bVal2020[12] / 2 + np.sum(bVal2020[13:]));

# As a proportion of the total:
vVal = plus50 / np.sum(bVal2020);

# Plotting the histogram.
# Data for 2020.
fig, ax = plt.subplots(dpi=144);
ax.hist(bins2020[:-1] + 2, bins2020, weights = bVal2020, 
        histtype = 'step', linewidth=0.4);

# The mean grade is displayed according to its position.
ax.text(mean20, np.max(bVal2020), f'2020 µ: {mean20:.3f}', 
        ha='right', va='center', fontsize=8);

# Standard deviation displayed below and in alignment with the mean.
ax.annotate(f'2020 σ: {sd20:.3f}', xy=(mean20, np.max(bVal2020) / 2 + 1),
            xytext=(mean20-sd20, np.max(bVal2020) / 2 + 1), 
            arrowprops=dict(arrowstyle="->", linewidth=0.7), 
            ha='right', va='center', fontsize=8);

# Data for 2024.
ax.bar(counts2024[0], height=counts2024[1]);

# Annotating the mean value.
ax.text(mean24, np.max(counts2024[1]) + 1, f'2024 µ: {mean24:.3f}', 
        ha='left', va='center', fontsize=8);

# Annotating the standard deviation from the mean.
ax.annotate(f'2024 σ: {sd24:.3f}', xy=(mean24, np.max(counts2024[1]) / 2),
            xytext=(mean24+sd24, np.max(counts2024[1]) / 2), 
            arrowprops=dict(arrowstyle="->", linewidth=0.7),
            ha='left', va='center', fontsize=8);

# Formatting.
ax.set_xticks(np.linspace(np.min(bins2020), np.max(bins2020), num=11));
ax.set_xticks(np.linspace(np.min(bins2020), np.max(bins2020), num=101), 
                          minor=True);
ax.set_yticks(np.linspace(0.00, 45, 46),
              minor=True);
ax.set_xlim(np.min(bins2020), np.max(bins2020));

ax.set_xlabel('Grade achieved');
ax.set_ylabel('Score frequency');

ax.grid(visible=True, which='both', axis='both',
        linestyle=':', linewidth=0.4, alpha=0.8);

ax.set_title('Class exam score distribution');
ax.legend(['2020 data', '2024 data']);

# Annotating the V-value.
ax.text(np.max(bins2020)-1, np.max(bVal2020)*0.75,
        f'V-value (proportion scoring 50\nand above in the 2020 exam)\n= {vVal:.5f}',
        ha='right', fontsize=6);

# Annotating Student ID.
ax.text(1, np.max(bVal2020), 'Student ID: 16044885', va='center', fontsize=8);

# Saving figure.
plt.savefig('16044885.png');