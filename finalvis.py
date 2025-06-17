# -*- coding: utf-8 -*-

# Importing pandas to convert the database to a dataframe.
import pandas as pd;
# Importing numpy for numerical operations/handling arrays.
import numpy as np;
# Matplotlib and seaborn necessary for the plotting.
import matplotlib.pyplot as plt;
import seaborn as sns;

# Multi-indexing by the country exporting and in which Series code.
data=pd.read_csv('Dataset\TB_burden_countries_2024-04-18.csv', 
                   index_col=['g_whoregion',
                              'year',                              
                              'iso3']).drop(labels=['country',
                                                    'iso2',
                                                    'iso_numeric'],
                                                    axis=1).sort_values(['g_whoregion',
                                                                        'iso3',
                                                                        'year']);
                                                                         
# Grouping by the mean average of countries within the WHO regional categories.
whoreg=data.groupby(['g_whoregion', 'year']).agg(np.mean);

def table22(data):
    """

    Parameters
    ----------
    data : Takes the WHO region categorised data to utilise the 2022 data and
    create a table for values.

    Returns
    -------
    None.

    """
    # Dropping superfluous year level of MultiIndex.
    data22=data.loc[(slice(None), [2022]), :].droplevel(1);
    # Creating a list of columns of interest to slice for the table.
    colint=['e_inc_100k', 'e_tbhiv_prct', 
            'e_mort_100k', 'e_mort_tbhiv_100k', 'cfr_pct', 'c_cdr'];
    intdata=data22.loc[:, colint].values;
    
    # Dictionary for the table columns' header row.
    coldic={
        'e_inc_100k' : 'Each incidence\nper 100k pop.',
        'e_tbhiv_prct' : 'HIV+ve in\nTB cases (%)', 
            'e_mort_100k' : 'All TB mortality\ncases\nper 100k pop.',
            'e_mort_tbhiv_100k' : 'HIV+ve in\nTB mortality\ncases\n'
            'per 100k pop.',
            'cfr_pct' : 'TB case\nfatality ratio(%)',
            'c_cdr' : 'TB case\ndetection rate/\ntreatment\ncoverage (%)'};
    
    # Creating a colormap to act as the background for the table cells.
    colbg = plt.cm.Reds(intdata / np.max(intdata));
    for x in np.arange(len(colint), dtype=int):
        colbg[:, x] = plt.cm.Reds(intdata[:, x] / 
                                  (1.5 * np.max(intdata[:, x])))
    colbg[:,-1] = plt.cm.Greens(intdata[:,-1] / 
                                (1.5 * np.max(intdata[:,-1])))
    
    collabs=[coldic[idx] for idx in colint];
    
    # Creation of the table.
    fig, ax=plt.subplots(dpi=144, layout='constrained');
    table=ax.table(cellText=np.round(intdata, decimals=3), cellColours=colbg,
                   cellLoc='center', 
                   rowLabels=data22.index, colLoc='center', colLabels=collabs,
                   bbox=[0,0,1,1], fontsize=16);
    table.auto_set_font_size(False);
    table.set_fontsize(8);
    ax.axis('off');
    ax.set_title('WHO tuberculosis burden estimates per WHO classified region:'
                 ' YR2022', fontsize=11);
    fig.savefig('table.png');
    
def hivmort(data):
    """

    Parameters
    ----------
    data : Takes the dataset and averages each year across countries for a
    pointplot timeseries of infections and mortality both positive and negative
    for HIV per 100k in the population.

    Returns
    -------
    None.

    """
    
    # Creating a list of columns of interest to slice for the pointplot.
    colint=['e_inc_100k', 'e_mort_100k', 'e_mort_tbhiv_100k'];
    colours=['orange', 'k', 'r'];
    intdata=data.loc[:, colint];
    coldic={
        'e_inc_100k' : 'Infection incidences',
            'e_mort_100k' : 'All mortality',
            'e_mort_tbhiv_100k' : 'HIV+ve mortality'};
    collabs=[coldic[idx] for idx in colint];
    
    # Creating the pointplot.
    fig, ax=plt.subplots(dpi=144, layout='constrained');
    for x in np.arange(len(colint)):
        sns.pointplot(intdata, x=intdata.index, y=intdata[colint[x]],
                      errorbar=('se'), color=colours[x], capsize=0.5,
                      label=collabs[x]);
    ax.grid(visible=True, which='both', linewidth=0.8, linestyle=':')
    # Changing transparency/size for better clarity.
    plt.setp(ax.collections, linewidth=1, alpha=0.2)
    plt.setp(ax.lines, linewidth=2, alpha=0.6)
    
    ax.set_yticks(np.linspace(0, stop=220, num=23), minor=True);
    ax.set_ylabel('Mean incidences per 100k', fontsize=10);
    ax.set_xlabel('Year of record', fontsize=10);
    ax.tick_params(axis='x', labelrotation=80);
    ax.tick_params(axis='both', labelsize=8);
    ax.set_title('Average incidences/mortalities across all WHO regions yearly',
                 fontsize=11);
    ax.legend(fontsize=8, loc='upper right')
    fig.savefig('hivmort.png')
    
def hivmortrat(data):
    """
    

    Parameters
    ----------
    data : Takes the dataset to plot a scatter graph of HIV+ve ratio against
    case fatality ratio.

    Returns
    -------
    None.

    """
    
    colint=['e_inc_100k', 'e_tbhiv_prct', 'cfr_pct'];
    intdata=data.loc[:, colint];
    intdata.rename(columns={'e_inc_100k' : 'Est. incidences\nper 100k', 
                          'e_tbhiv_prct' : 'Est. TB/HIV+ve percentage',
                          'cfr_pct' : 'Case fatality ratio'}, inplace=True);
    intdata.rename_axis(index={'g_whoregion' : 'WHO Region'}, inplace=True);
    
    # Plotting the scatter graph.
    ax=sns.relplot(data=intdata, x='Est. TB/HIV+ve percentage', 
                   y='Case fatality ratio', kind='scatter', 
                   size='Est. incidences\nper 100k', hue='WHO Region', 
                   alpha=0.3, marker='x', legend='auto');
    ax.set_axis_labels('Percentage (%) of cases HIV+ve', 
                       'Percentage (%) case fatality ratio', 
                       fontsize=10).set(xscale='log', yscale='log');
    plt.title(label='Percentage (%) of TB cases testing HIV+ve\n'
             'against case fatality ratio of the same year', fontsize=11);
    plt.grid(visible=True, which='both', linewidth=0.8, linestyle=':');
    ax.savefig('hivfatrat.png', dpi=144);
    
def trtmort(data):
    """
    

    Parameters
    ----------
    data : Takes the AFR data to plot treatment rates overlaid with mortality
    rates as histograms.

    Returns
    -------
    None.

    """
    
    # Slicing case detection rate/case fatality percent.
    intdata=data.loc[:, ['c_cdr', 'cfr_pct']];
    
    # Plotting the fatality rate over detection rate to highlight any correlation.
    fig, ax=plt.subplots(dpi=144, layout='constrained');
    ax.bar(intdata.index, intdata['c_cdr'], label='Case detection\nrate (%) /'
                            '\nTreatment coverage');
    ax.bar(intdata.index, intdata['cfr_pct'], label='Case fatality '
                                                   'percentage (%)',
                                                   width=0.5);
    
    # Aesthetics/formatting
    ax.set_yticks(np.arange(0, 70, step=1), minor=True);
    ax.set_ylabel('Percentage of cases');
    ax.set_xticks(np.arange(np.min(intdata.index), np.max(intdata.index) + 1));
    ax.set_xlabel('Year of record');
    ax.set_xlim(1999.5, 2022.5);
    ax.tick_params(axis='both', labelsize=10);
    ax.tick_params(axis='x', labelrotation=80);
    ax.grid(axis='y', which='both', linestyle='--', linewidth=0.3);
    ax.set_title('Estimated case fatality % overplotted on case detection'
                 ' rate\nfor WHO region AFR', fontsize=10);
    ax.legend(fontsize=8);
    fig.savefig('trtmort.png', dpi=144);
    
table22(whoreg);
hivmort(whoreg.droplevel(0));
hivmortrat(data);
trtmort(whoreg.loc['AFR']);