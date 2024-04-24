# -*- coding: utf-8 -*-

# Importing pandas to convert the database to a dataframe.
import pandas as pd;

# Importing numpy for numerical operations/handling arrays.
import numpy as np;

# Importing curve_fit from scipy for fitting function.
from scipy.optimize import curve_fit;

# Importing KMeans and silhouette score for clustering.
from sklearn.cluster import KMeans;
from sklearn.metrics import silhouette_score;

# Importing RobustScaler to rescale the data as the data is not
# normally distributed.
from sklearn.preprocessing import RobustScaler;

# Matplotlib for plotting.
import matplotlib.pyplot as plt;
import seaborn as sns;

reds=pd.read_csv('Dataset/winequality-red.csv', delimiter=';');

def cnfmat(data):
    """
    
    Parameters
    ----------
    data : Takes the dataset and produces a confusion matrix plot to highlight
    correlation between different attributes of the wine variants.

    Returns
    -------
    None.

    """
    
    # Initialising axes and figure.
    fig, ax = plt.subplots(dpi=144, layout='tight');
    
    # Creating a Pearson correlation matrix & mask.
    comat = data.corr();
    ax.imshow(comat, cmap='plasma');
    ax.set_xticks(np.arange(0, len(comat.columns)));
    ax.set_yticks(np.arange(0, len(comat.columns)));
    ax.set_xticklabels(comat.columns,  fontsize=7,
                       ha='right', rotation=35, rotation_mode='anchor');
    ax.set_yticklabels(comat.columns, fontsize=7);
    ax.set_title('Correlation between attributes of red wine variants',
                 fontsize=10, ha='center')
    
    for i in range(len(comat.columns)):
        for j in range(len(comat.columns)):
            ax.text(j, i, round(comat.loc[comat.columns[i],
                                          comat.index[j]], 1), 
                    ha='center', va='center', color='white',
                    size=8);
    for k in range(len(comat.columns)):
        ax.text(k, k, round(comat.loc[comat.columns[k], 
                                      comat.index[k]], 1),
                ha='center', va='center', color='k',
                size=8);
    fig.savefig('cnfmat.png', dpi=144);
    return;
    
def linfunc(x, m, c):
    """
    
    Parameters
    ----------
    x : x data.
    m : gradient of slope.
    c : intercept at x = 0.

    Returns
    -------
    y : function for use in fitting methods.

    """
    y = m * x + c;
    return y;
    
def fit_line(data):
    """
    
    Parameters
    ----------
    data : Red wine data for alcohol content and quality.
    
    Fits the line to the distribution.

    Returns
    -------
    optparam : Optimal parameters to minimize sum of squared residuals.
    cov : Estimated covariance for optparam.
    sig : The errorbars' distance from the line.

    """
    
    p, cov = curve_fit(linfunc, data['alcohol'], data['quality']);
    sig = np.sqrt(np.diag(cov));
    return p, cov, sig;
    
def bars_with_fit(data, desc):
    """
    
    Parameters
    ----------
    data : Red wine alcohol/quality data.
    desc : Dataframe containing the first four moments of the distribution.

    Plots a seaborn histogram with overplotted line fitted to the distribution.

    Returns
    -------
    None.

    """
    # Assigning statistical moments to variables.
    mean = desc.loc['alcohol', 'mean'];
    std = desc.loc['alcohol', 'std'];
    skw = desc.loc['alcohol', 'skew'];
    krt = desc.loc['alcohol', 'kurtosis'];
    
    # Range for the ticks & labels.
    xrange=np.linspace(np.min(data['alcohol']), 
                              np.max(data['alcohol']), num=20)
    xlab=np.round(xrange, 1);
    
    # Initialise and plot the bars.
    fig, ax = plt.subplots(dpi=144, layout='constrained');
    sns.histplot(data, x='alcohol', y='quality', bins=len(xrange) * 5, ax=ax);
    
    # Printing statistical moments on the plot
    ax.text(mean, 10, f'µ: {mean:.3f}', 
            ha='center', va='top', fontsize=8);
    ax.axvline(mean, 0.65, 0.85, color='k', linewidth=0.5)
    ax.annotate(f'σ: {std:.3f}',(mean, 7.5), (mean+std, 7.5),
                ha='left', va='center', arrowprops=dict(arrowstyle='-',
                                                        linewidth=0.5,
                                                        shrinkB=0), 
                fontsize=8);
    ax.text(mean * skw, 6.5, f'skew: {skw:.3f}', 
            va='center', fontsize=8);
    ax.text(mean + 0.1, 8.5, f'kurtosis: {krt:.3f}', 
            ha='left', va='center', fontsize=8);
    
    # Plot the fitted line with errorbars.
    p, cov, sig = fit_line(data);
    ax.plot(xrange, linfunc(xrange, p[0], p[1]), color='k',
            linestyle='--', linewidth=0.5);
    ax.fill_between(xrange, 
                    linfunc(xrange, p[0]-sig[0], p[1]-sig[1]),
                    linfunc(xrange, p[0]+sig[0], p[1]+sig[1]),
                    color='red',
                    linestyle=':', linewidth=0.5, alpha=0.08);
    
    # Formatting.
    ax.set_xticks(xrange);
    ax.set_xticklabels(labels=xlab, ha='right', va='center', 
                       rotation=35, rotation_mode='anchor');
    ax.set_yticks(np.arange(1, 11, step=1));
    ax.set_ylim(1, 11);
    ax.set_xlabel('Alcohol content', fontsize=11);
    ax.set_ylabel('Quality of wine', fontsize=11);
    ax.set_title('Least squares method fit of alcohol content\n'
                 'against quality of red wine', fontsize=12);
    fig.savefig('barfit.png', dpi=144);
    return;
                                      
def clust(n, data, scaler, wcss):
    """
    
    Parameters
    ----------
    n : Number of clusters.
    data : Compared variables.
    scaler : Instance of RobustScaler()
    wcss = Within-Cluster Sum of Square values.
    
    Calculates the silhouette score / 
    Within-Cluster Sum of Square value for 'n' clusters and returns them.

    Returns
    -------
    score = Silhouette score.
    wcss = Within-Cluster Sum of Square values.
    labels = Labels of cluster centres.
    kmeans = KMeans instance.
    cent = Backscaled estimated cluster centres' coordinates.
    scaler : Instance of RobustScaler()

    """
    # Passes the number of clusters to compute the score for.
    kmeans = KMeans(n_clusters=n, n_init=16);
    
    # Fittting the data.
    kmeans.fit(data);
    labels = kmeans.labels_;
    cent = scaler.inverse_transform(kmeans.cluster_centers_);
    
    # Appending the wcss array for the current value of clusters.
    wcss.append(kmeans.inertia_);
    
    # Calculating the silhouette score to return.
    score = silhouette_score(data, labels);
    
    return score, wcss, labels, kmeans, cent;
    
def bestclst(data, scaler):
    """
    
    Parameters
    ----------
    data : Red wine data quality and alcohol content.
    scaler : Instance of RobustScaler()
    
    Iterates over the range of clusters and checks if the current
    silhouette score is greater than the current best, replacing it if so.
    
    Returns
    -------
    best_n : Best number of clusters 'n' based on the greatest silhouette score
            compared with elbow plot defined later.
    wcss : Within-Cluster Sum of Square.
    labels = Labels of cluster centres.
    scaler : Instance of RobustScaler()
    best_score : Highest silhouette score.

    """
    
    # Initialises the variables to hold the highest score and corresponding 'n'
    # of clusters.
    best_n, best_score = None, -np.inf;
    wcss = [];
    
    # Range of clusters =  2 -> 1% of samples.
    for n in range(2, int(len(data.index) / 100) + 2):
        # RobustScaler applied to transform the data before its silhouette 
        # scores / WCSS are calculated.
        scl = scaler.fit_transform(data)
        score, wcss, labels, kmeans, cent = clust(n, scl, scaler, wcss);
        if score > best_score:
            best_n = n;
            best_score = score;
        print(f'{n} clusters silhouette score: {score}');
        
    print(f'Best score = {best_score}\nfor {best_n} clusters');
    return best_n, wcss, scl, scaler, labels, best_score;

def elbow(kmin, kmax, wcss, silclst, silscr):
    """
    
    Parameters
    ----------
    kmin : Lowest number of clusters.
    kmax : 1/100th of all samples as number of clusters.
    wcss : Within-Cluster Sum of Square.
    silclst : The best number of clusters according to prior silhoutte score.
    silscr : Highest silhouette score.

    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots(dpi=144, layout='constrained');
    ax.plot(np.arange(kmin, kmax, step=1), wcss, 
            'k+-', linewidth=0.5);
    ax.set_xticks(np.arange(kmin, kmax, step=1));
    ax.set_xlim(kmin - 1, kmax - 1);
    ax.scatter(silclst, wcss[silclst-kmin], s=100, c='r', marker='x', 
               label='Silhouette score identified clusters');
    ax.text(silclst + 0.5, wcss[silclst-kmin], f'Silhouette score: {silscr}',
            fontsize=8, va='center');
    ax.set_xlabel('N clusters', fontsize=11);
    ax.set_ylabel('Within-Cluster Sum of Square', fontsize=11);
    ax.tick_params(labelsize=8);
    ax.set_title('Elbow plot: WCSS against number of clusters', fontsize=12);
    ax.legend();
    fig.savefig('elbow.png', dpi=144);
    return;
    
def sulphac(data, scaler, silclst):
    """
    
    Parameters
    ----------
    data : The red wine sample dataset sliced to cluster by Total Sulphur
    Dioxide / Fixed Acidity.
    scaler : RobustScaler() instance used as distribution is not normal.
    silclst : The best number of clusters based on silhouette score
    corroborated by the elbow plot.

    Plots the clusters of wine samples categorised by sulphur dioxide quantity
    and acidity of the sample.

    Returns
    -------
    cluster : The cluster number which corresponds to the sample.

    """
    
    scl = scaler.fit_transform(data);
    
    # Retrieving kmeans and labels from the function for a plot.
    score, wcss, labels, kmeans, cent = clust(silclst, scl, scaler, []);
    print(kmeans);
        
        
    # Variable holding the cluster centres' coordinates.
    xkmeans = cent[:,0];
    ykmeans = cent[:,1];
        
    # Holds the estimate of which cluster each coordinate belongs to.
    centlabs = kmeans.predict(kmeans.cluster_centers_);
        
    # Plotting a scatter of the alcohol/quality data.
    fig, ax = plt.subplots(dpi=144, layout='constrained');
    ax.scatter(data['total sulfur dioxide'], data['fixed acidity'], 
               c=labels + 2, cmap='tab20b', s=10, marker='o', 
                       label='Data', alpha=0.05);
    # Overplotting the cluster centre estimates on the same axes.
    paths = ax.scatter(xkmeans, ykmeans, c=centlabs, cmap='tab20b', 
                       marker='1', s=200, linewidth=1.4, 
                       label='Estimated cluster centres');
    
    # Creating a colorbar which indicates the different clusters for
    # each colour.
    colbar = fig.colorbar(paths, ax=ax);
    colbar.set_ticks(np.unique(labels));
    colbar.set_ticklabels(np.unique(labels) + 1);
    
    # Formatting of axes/ticks.
    ax.set_title('Clusters of wine samples formed on acidity/\n'
                 'sulphur dioxide relationship', fontsize=12);
    xrange = np.linspace(np.min(data['total sulfur dioxide']), 
                              np.max(data['total sulfur dioxide']),
                              num=10);
    ax.set_xticks(xrange)
    ax.set_xticklabels(np.round(xrange, 1), rotation=35, ha='right');
    ax.set_xlabel('Total Sulphur Dioxide', fontsize=11);
    
    yrange = np.linspace(np.min(data['fixed acidity']), 
                              np.max(data['fixed acidity']),
                              num=10);
    ax.set_yticks(yrange);
    ax.set_yticklabels(np.round(yrange, 1));
    ax.set_ylabel('Fixed Acidity', fontsize=11);
    ax.legend();
    
    fig.savefig('clust.png', dpi=144);
    
    return labels + 1;
    
def violin(data):
    """
    
    Parameters
    ----------
    data : Red wine variants' data.
    
    Creates a violin plot comparing the 3 determined clusters by quality with
    respect to the alcohol content.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(dpi=144, layout='constrained');
    sns.violinplot(data, x='quality', y='alcohol', hue='cluster',
                   palette='Dark2', inner='quart', 
                   linewidth=0.8, ax=ax, cut=0);
    
    # Formatting axes/ticks.
    yrange = np.linspace(np.min(data['alcohol']), 
                              np.max(data['alcohol']),
                              num=10);
    ax.set_yticks(yrange);
    ax.set_yticklabels(np.round(yrange, 1));
    ax.grid(axis='y', which='major', linestyle=':', linewidth=0.4);
    ax.set_xlabel('Quality of red wine', fontsize=11);
    ax.set_ylabel('Alcohol content', fontsize=11);
    ax.set_title('Alcohol content by quality of red wine,'
                 '\nseparated by established cluster', fontsize=12)
    
    fig.savefig('violin.png', dpi=144);
    
    return;
   
# Calculating the standard deviation for each column of the dataset.
desc = reds.describe().transpose();
desc['skew'] = reds.skew();
desc['kurtosis'] = reds.kurtosis();
print(desc);

for x in reds.columns:
    print(f'{x} standard deviation: {np.std(reds[x])}');

# Confusion matrix to highlight correlation and assist when deciding which
# attributes to determine clustering around.
cnfmat(reds);

bars_with_fit(reds[['alcohol', 'quality']], desc.loc[['alcohol']]);

# Creating instance of scaler - RobustScaler() as distribution is not normal.
scaler = RobustScaler();

# Function to return the best cluster/silhouette score
silclst,wcss,scl,scaler,labels,silscr = bestclst(reds[['total sulfur dioxide',
                                                       'fixed acidity']],
                                                 scaler);

elbow(2, int(len(reds.index) / 100) + 2, wcss, silclst, silscr);

reds['cluster'] = sulphac(reds[['total sulfur dioxide', 'fixed acidity']],
                          scaler, silclst);

# Creating a violin plot 
violin(reds[['alcohol', 'quality', 'cluster']]);