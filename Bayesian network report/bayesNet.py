# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 08:56:10 2024

Bayesian Networks code
"""

import numpy as np;
import pandas as pd;
import pymc as pm;
import arviz as az;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.model_selection import train_test_split;
from imblearn.over_sampling import SMOTE;
from sklearn.tree import DecisionTreeClassifier, plot_tree;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.metrics import classification_report, confusion_matrix;
from sklearn.metrics import roc_auc_score, roc_curve;



def bayesnet(X, y):
    
    """
    Bayesian network creation

    Takes the training data and labels to fit a bayesian network to the data with
    weakly informative priors on account of the unknown relationship between the
    dataset's features except for the assumption that they in some manner influence
    the classification.

    Returns a trace of a sample taken from the model, as well as the model itself.
    """
    
    with pm.Model() as model:
        
        # Priors for the regression coefficients, each distribution corresponds
        # to a feature of the dataset.
        alpha = pm.Normal('alpha', mu=0, sigma=10);
        beta_tc = pm.Normal('beta_tc', mu=0, sigma=10);
        beta_sz = pm.Normal('beta_sz', mu=0, sigma=10);
        beta_sh = pm.Normal('beta_sh', mu=0, sigma=10);
        beta_ma = pm.Normal('beta_ma', mu=0, sigma=10);
        beta_ec = pm.Normal('beta_ec', mu=0, sigma=10);
        beta_bn = pm.Normal('beta_bn', mu=0, sigma=10);
        beta_bc = pm.Normal('beta_bc', mu=0, sigma=10);
        beta_nn = pm.Normal('beta_nn', mu=0, sigma=10);
        beta_mi = pm.Normal('beta_mi', mu=0, sigma=10);

        # Linear combination of inputs.
        mu = alpha + beta_tc * X['Clump thickness'] \
            + beta_sz * X['Uniformity of Cell Size'] \
            + beta_sh * X['Uniformity of Cell Shape'] \
            + beta_ma * X['Marginal Adhesion'] \
            + beta_ec * X['Single Epithelial Cell Size'] \
            + beta_bn * X['Bare Nuclei'] \
            + beta_bc * X['Bland Chromatin'] \
            + beta_nn * X['Normal Nucleoli'] \
            + beta_mi * X['Mitoses'];

        # Likelihood (using the logit link function for binary outcome)
        p = pm.Deterministic('p', pm.math.invlogit(mu));
        likelihood = pm.Bernoulli('likelihood', p=p, observed=y);
        
        # Sampling inference data.
        trace = pm.sample(1000, tune=1000, cores=1, chains=4,
                          target_accept=0.99, 
                          random_seed=RAND); # Experienced issues running multiple cores.
        
        postpredsamp = pm.sample_posterior_predictive(trace, random_seed=RAND);
        
        # Plotting inferences.
        az.plot_trace(trace, figsize=(12, 36), 
                      backend_kwargs={'dpi' : 144}, 
                      compact=True);
        fig, ax = plt.subplots(dpi=144);
        az.plot_ppc(postpredsamp, num_pp_samples=100, 
                    ax=ax);
        ax.set_title('Fig 2. Posterior/prior predictive checks');
        
        
    
    return(trace, model);



def bayesPred(X, post, model):
    
    """
    Performing predictions with the Bayesian network model

    Takes the test data from the same split as used to predict with the decision
    tree and forest as well as the posteriors from the sample's trace and the model
    that produced said trace to predict the class labels for the test data.
    """
    
    with model:

        # Linear combination of inputs.
        mu = trace.posterior['alpha'].mean() + trace.posterior['beta_tc'].mean() * X[0] \
            + trace.posterior['beta_sz'].mean() * X[1] \
            + trace.posterior['beta_sh'].mean() * X[2] \
            + trace.posterior['beta_ma'].mean() * X[3] \
            + trace.posterior['beta_ec'].mean() * X[4] \
            + trace.posterior['beta_bn'].mean() * X[5] \
            + trace.posterior['beta_bc'].mean() * X[6] \
            + trace.posterior['beta_nn'].mean() * X[7] \
            + trace.posterior['beta_mi'].mean() * X[8];

        # Prediction for labels.
        predict = 1 / (1 + np.exp(-mu));
        return(float(predict));
    


def bayesPredLab(idx):
    
    """
    Wrapper for the bayesPred function
    """
    
    return(bayesPred(idx, post, model));



def forest(X_train, X_test, y_train, y_test, osLabel):
    
    """
    Decision tree & random forest creation

    Fits a model to predict values using the decision tree and random forest
    classifiers.  
    Plots the structure of the decision tree.
    Plots the true positive rate against the false positive rate for both the tree
    and random forest.

    Takes the training/test split variables as well as a marker to indicate whether
    the data is oversampled using the SMOTE oversampling method.
    """
    
    # Fitting the decision tree.
    decTree = DecisionTreeClassifier(random_state=RAND);
    decTree.fit(X_train, y_train);
    
    # Drawing predictions.
    pred = decTree.predict(X_test);
    print(f'{osLabel}: Decision Tree classification');
    decRep = classification_report(y_test, pred,
                                   output_dict=True);
    decRep = pd.DataFrame.from_dict(decRep).transpose();
    print(decRep);
    DTconfmat = confusion_matrix(y_test, pred);
    
    # Plotting the receiver operating characteristic & area under ROC.
    decTree_roc_auc = roc_auc_score(y_test, pred);
    falsePR, truePR, thresholds = roc_curve(y_test, 
                                            decTree.predict_proba(X_test)[:,1]);
    
    # Visualisation of model.
    features = list(data.columns[:-1]);
    plt.figure(figsize=(18,6), dpi=144, layout='constrained');
    plot_tree(decTree, feature_names=features, filled=True, rounded=True,
              fontsize=6);

    # Fitting a random forest classifier.
    rfc = RandomForestClassifier(n_estimators=100, random_state=RAND);
    rfc.fit(X_train, y_train);
    
    # Drawing predictions.
    rfc_pred = rfc.predict(X_test);
    print(f'{osLabel}: Random Forest classification');
    rfcRep = classification_report(y_test, rfc_pred,
                                   output_dict=True);
    rfcRep = pd.DataFrame.from_dict(rfcRep).transpose();
    print(rfcRep);
    RFCconfmat = confusion_matrix(y_test, rfc_pred);
    
    # Plotting confusion matrices for decision trees and RFC with classification report.
    fig, ax = plt.subplots(dpi=144);
    ax = sns.heatmap(DTconfmat, cmap='RdYlBu', 
                     annot=True, annot_kws={'size':'x-large'});
    ax.set_title(f'Fig 3. Actual classification/Predicted classification:\nDecision tree - {osLabel}');
    ax.set_xticks([0.5,1.5], ['Benign', 'Malignant']);
    ax.set_yticks([0.5,1.5], ['Benign', 'Malignant']);
    ax.set_ylabel('Actual class');
    ax.set_xlabel('Predicted class');
    ax.table(cellText=decRep.values.round(decimals=3), rowLabels=decRep.index,
                colLabels=decRep.columns, bbox=[0.2, -0.7, 1, 0.5]);
    
    fig, ax = plt.subplots(dpi=144);
    ax = sns.heatmap(RFCconfmat, cmap='RdYlBu', 
                     annot=True, annot_kws={'size':'x-large'});
    ax.set_title(f'Fig 4. Actual classification/Predicted classification:\nRandom forest - {osLabel}');
    ax.set_xticks([0.5,1.5], ['Benign', 'Malignant']);
    ax.set_yticks([0.5,1.5], ['Benign', 'Malignant']);
    ax.set_ylabel('Actual class');
    ax.set_xlabel('Predicted class');
    ax.table(cellText=rfcRep.values.round(decimals=3), rowLabels=rfcRep.index, 
                colLabels=rfcRep.columns, bbox=[0.2, -0.7, 1, 0.5]);
    
    # Plotting the receiver operating characteristic & area under ROC.
    rfc_roc_auc = roc_auc_score(y_test, rfc_pred);
    rfFPR, rfTPR, RFthresholds = roc_curve(y_test, 
                                            rfc.predict_proba(X_test)[:,1]);
    fig, ax = plt.subplots(dpi=144, layout='tight');
    ax.plot(falsePR, 
            truePR, 
            label = f'Decision tree (ROC area under curve: {decTree_roc_auc:.2f}',
            linewidth = 1);
    ax.plot(rfFPR, 
            rfTPR, 
            label = f'Random forest (ROC area under curve: {rfc_roc_auc:.2f}',
            linewidth = 1);
    ax.plot([0,1], [0,1], 'r--', linewidth = 1);
    ax.set_xlim([-0.01, 1.01]);
    ax.set_ylim([-0.01, 1.01]);
    ax.set_xlabel('False positive rate');
    ax.set_ylabel('True positive rate');
    for lc in ['top','right','bottom','left']:
        ax.spines[lc].set_visible(False);
    ax.set_title(f'Fig 5. {osLabel}: Receiver operating characteristic');
    ax.legend(loc = 'lower right');
    
    return();


"""
Data loading & preprocessing.

Dataset: 

Breast Cancer (UCI), classification -
https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)

Preprocessing:
    
Replaced NaN (?) values in the 'Bare_nuclei' column of the data features with
the mode of the column.

Altered the labelling of 
benign/malignant tumours to 0/1 respectively. (From 2/4)
"""

# Setting random seed value.
RAND = 26;

# Load dataset
data = pd.read_csv('data/breast+cancer+wisconsin+original/breast-cancer-wisconsin.data',
                   sep = ',', header = None,
                   names = ('Sample code number', 'Clump thickness',
                   'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size',
                   'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
                   'Mitoses', 'Class:'),
                   index_col = 0);


# Replacing NaN values for bare nuclei with a value of 0.
data['Bare Nuclei'] = data['Bare Nuclei'].replace(to_replace='?', \
                                                  value=data['Bare Nuclei']
                                                  .mode()[0]).astype(np.int64);

# Replacing the values 2 & 4 with 0 and 1 for benign/malignant.
BMdict = {
    2: 0,
    4: 1};
data.loc[:,'Class:'] = data.loc[:,'Class:'].replace(to_replace = BMdict);

X = data.loc[:,:'Mitoses'];


N, D = X.shape;
D0 = int(D / 2);

y = data.loc[:,'Class:'];


# Splitting training & testing data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=RAND);


# Oversampling training data via SMOTE to balance the malignant/benign data ratio.
osamp = SMOTE(random_state=26);
columns = X_train.columns;

osamp_dataX, osamp_datay = osamp.fit_resample(X_train, y_train);
osamp_dataX = pd.DataFrame(osamp_dataX, columns=columns);


forest(X_train, X_test, y_train, y_test, 'No oversampling');
forest(osamp_dataX, X_test, osamp_datay, y_test, 'SMOTE oversampling');


# Training Bayesian network and printing summary of the trace returned.
trace, model = bayesnet(X_train, y_train);
print(pm.summary(trace));


cats = ['alpha', 'beta_tc', 'beta_sz', 'beta_sh', 'beta_ma', 'beta_ec', 'beta_bn',
        'beta_bc', 'beta_nn', 'beta_mi'];
post = np.zeros(len(cats));

for c in np.arange(len(cats)):
    post[c] = trace.posterior[cats[c]].mean();
    

# Appending a predictions column onto the test data it predicts from.
X_test['predict'] = X_test.apply(bayesPredLab, axis=1);

# For the purposes of easy comparison between predicted and real observation.
X_test['actual'] = y_test;

# Plotting a confusion matrix of the actual labels against Bayesian prediction.
# Plotting a classification report below the matrix in addition.
bayConfmat = confusion_matrix(X_test['actual'], np.round(X_test['predict']));
bayRep = classification_report(X_test['actual'], np.round(X_test['predict']),
                               output_dict=True);
bayRep = pd.DataFrame.from_dict(bayRep).transpose();
fig, ax = plt.subplots(dpi=144);
ax = sns.heatmap(bayConfmat, cmap='RdYlGn', 
                 annot=True, annot_kws={'size':'x-large'});
ax.set_title('Fig 1. Actual classification/Predicted classification:\nBayesian network');
ax.set_xticks([0.5,1.5], ['Benign', 'Malignant']);
ax.set_yticks([0.5,1.5], ['Benign', 'Malignant']);
ax.set_ylabel('Actual class');
ax.set_xlabel('Predicted class');
ax.table(cellText=bayRep.values.round(decimals=3), rowLabels=bayRep.index,
            colLabels=bayRep.columns, bbox=[0.2, -0.7, 1, 0.5]);

