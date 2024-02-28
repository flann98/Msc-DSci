# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 19:17:04 2024

@author: brend
""" 

# Function imports.

import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;


def plot_chn_pop(chndata):
    
    """
    This function takes the data read after the function imports and uses only
    the data for the population counts of rural & urban China to plot a 
    comparison of the demographic changes therein over time.
    
    """    
    
    # Following this comment is the code for the chinese population data plot,
    # which slices from the a CHN country/identification code multi index.
    pd.MultiIndex.from_frame(chndata);
    chnpopdist=chndata.loc[[('CHN', 'Rural population'), ('CHN', 'Urban population')],
                           '1960':].transpose();
    
    # Creates chnpopmils variable to hold a DataFrame where the population is
    # counted by the millions.
    mlldiv = 10**-6
    chnpopmils = chnpopdist * mlldiv;
    
    # Initialising the plot & axes labels/limits.
    fig, ax = plt.subplots();
    fig.set_dpi(144);
    ax.plot(chnpopmils, label=("Rural population","Urban population"));
    ax.set_ylabel("Population (millions)");
    ax.set_xlabel("Year");
    ax.set_xlim("1960", "2022");
    ax.set_ylim(np.min(chnpopmils), np.max(chnpopmils));
    
    # Plotting a line perpendicular to the year to illustrate the point at
    # which a correlation is measured from.
    ax.vlines(40, np.min(chnpopmils), np.max(chnpopmils), color='k', 
              linestyles='dashdot', linewidth=0.9);
    
    # Adding the correlation coefficient as text with an arrow indicating
    # direction.
    ax.text(51, chnpopmils.iloc[40][('CHN','Urban population')], 
            ('Correlation:\n' 
             + chnpopmils.iloc[40:].corr().iloc[1][0].astype(str)), ha='center',
            va='top');
    ax.annotate('', xy=(61, chnpopmils.iloc[40][('CHN','Urban population')] - 100),
                xytext=(41, chnpopmils.iloc[40][('CHN','Urban population')] - 100),
                arrowprops=dict(arrowstyle="->"));
    
    # For the y axis, sets a range from the smallest recorded population in the
    # DataFrame to the largest and sets the tick steps to be proportional to it.
    plt.xticks(np.arange(0, 61, step=4), rotation=60, size=8);    
    plt.yticks(ticks=np.arange(np.min(chnpopmils), np.max(chnpopmils)+0.1,
                               step=(np.max(chnpopmils) - np.min(chnpopmils)) / 14),
               size=8);
    plt.grid(visible=1, linestyle='--', linewidth=0.5);
    plt.title("Demographic changes in rural/urban China from 1960 to 2022", size=10);
    plt.legend(loc='lower right', fontsize=8);
    plt.show();
    
    return;
    

def plot_lit(data):
    """
    This function takes the data read after the function imports and uses only
    the data for the literacy rates of multiple countries to plot a 
    comparison of the change from one year to another as a bar chart.
    
    """
    
    pd.MultiIndex.from_frame(data);
    ltcdata=data.loc[[('LIC',
            'Literacy rate, adult total (% of people ages 15 and above)'),
            ('LIC', 'Literacy rate, youth total (% of people ages 15-24)'),
            ('MIC', 'Literacy rate, adult total (% of people ages 15 and above)'),
            ('MIC', 'Literacy rate, youth total (% of people ages 15-24)'),
            ('WLD', 'Literacy rate, adult total (% of people ages 15 and above)'),
            ('WLD', 'Literacy rate, youth total (% of people ages 15-24)')],
            ('1990', '2000', '2010', '2020')];
    
    
    # ltcmom holds the major moments.
    ltcmom=ltcdata.describe().transpose();
    
    # Utilising pandas functions to add skewness and kurtosis to ltcmom.
    ltcmom['skew']=ltcdata.skew();
    ltcmom['kurtosis']=ltcdata.kurtosis();    

    # Transpose to have the years read off as the indexes, then plots a bar chart.
    ax1=ltcdata.transpose().plot(kind='bar');
    ax1.set_title("Percentage literacy between countries over time");
    ax1.set_xlabel("Year");
    ax1.set_ylabel("Percentage literacy in population");
    ax1.set_xlim(-0.5, 4);
    ax1.set_ylim(0, 100);
    ax1.set_xticks(ticks=np.arange(0, 4, step=1));
    ax1.set_yticks(ticks=np.arange(0, 100, step=10), minor='True');
    ax1.grid(visible='True', axis='y', which='both', linestyle='--', linewidth=0.3);
    ax1.legend(loc='lower right', fontsize=8);
    
    # Retrieves the figure and sets its resolution to 144dpi.
    plt.gcf().set_dpi(144);
    for x in range(len(ltcmom.index)):
        ax1.hlines(ltcmom['50%'][x], x-0.5, x+0.25, linestyles='dashdot', linewidth=0.4, color='k');
    for c in range(len(ltcmom.index)):
        ax1.text(c + .28, ltcmom['50%'][c], ('50%: '+round(ltcmom['50%'][c], 2).astype(str)),
                 fontsize=8, va='center');
        ax1.text(c, (ltcmom['max'][c]) + 1.7, ('$\sigma$: '+round(ltcmom['std'][c], 2).astype(str)),
                 fontsize=8);
        ax1.text(c + .28, (ltcmom['max'][c]) * 0.65, ('Skew:\n'+round(ltcmom['skew'][c], 2).astype(str)),
                 fontsize=8);
        ax1.text(c + .28, (ltcmom['max'][c]) * 0.5, ('Kurtosis:\n'+round(ltcmom['kurtosis'][c], 2).astype(str)),
                 fontsize=8);
    
    
    return();
    

def plot_hlthed(data):
    """
    This function takes the data read after the function imports and uses only
    the data for health/education of the years 2008 and 2018 to plot a 
    comparison as a heatmap.
    
    """
    pd.MultiIndex.from_frame(data);
    hlthdata=data.loc[[('WLD',
            'Adults (ages 15+) and children (0-14 years) living with HIV'),
            ('WLD', 'AIDS estimated deaths (UNAIDS estimates)'),
            ('WLD', 'Domestic private health expenditure per capita, PPP  (current international $)'),
            ('WLD', 'Fertility rate, total (births per woman)'),
            ('WLD', 'Life expectancy at birth, total (years)'),
            ('WLD', 'Out-of-pocket expenditure per capita, PPP (current international $)'),
            ('WLD', 'Public spending on education, total (% of GDP)'),
            ('WLD', 'School enrollment, tertiary (% gross)')], '2008':'2018'].transpose();
    hlthcorr=hlthdata.corr();
    fig2, ax2 = plt.subplots();
    fig2.set_dpi(144);
    ylabel=['0. ', '1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. '];
    
    # Appending numbers to what will be the ytick labels such that correspond
    # with labels on the x axis.
    for y in range(len(ylabel)):
        ylabel[y] += hlthcorr.columns.droplevel('Country Code')[y];
    ax2.set_xticks(np.arange(len(hlthcorr.columns)));
    ax2.set_yticks(np.arange(len(hlthcorr.columns)), labels=ylabel, fontsize=8);
    
    # Plotting the colour map.
    ax2.imshow(hlthcorr, cmap='RdYlGn');
    ax2.set_title('Worldwide correlation between changes in health & education related factors between 2008 and 2018',
                  loc='right', size=11);
    
    for i in range(len(hlthcorr.columns)):
        for j in range(len(hlthcorr.columns)):
            ax2.text(j, i, round(hlthcorr.loc[hlthcorr.columns[i],hlthcorr.index[j]], 3),
                     ha="center", va="center", color="k", size=8);
    return;


def plot_data(data):

    """
    Main function of the program, takes the previously defined functions and 
    executes them.
    
    """    
    
    plot_chn_pop(data);
    plot_lit(data);
    plot_hlthed(data);
    return;
    

data=pd.read_csv("Data/HNP_StatsData.csv", index_col=['Country Code','Indicator Name']);

plot_data(data);