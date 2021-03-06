
"""
Created on Fri Aug 27 15:30:05 2021

@author: oliver
"""

from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
import numpy as np

def ROC_custom(path,title,savepath):
    from scipy import stats
    #retruns:  auroc low value CI, high value CI
    name = path.split('.pkl')
    name = name[0].split('model')
    full_path = os.path.join(path, str(path))
    data = pd.read_csv(full_path)
    target_labelDict = {'MSIH': 0, 'nonMSIH': 1}
    #target_labelDict = {'MUT': 0, 'WT': 1}
    resultsPath = data
    
    try:
        y_true = list(data['y_true'])
        data_mut =data.loc[data['y_true'] == 1, 'nonMSIH']
        data_wt =data.loc[data['y_true'] == 0, 'MSIH']
        stats_value , p_value = stats.ttest_ind(data_mut, data_wt)
    except KeyError:
        y_true = list(data['y_1'])
        data_mut =data.loc[data['y_1'] == 1, 'nonMSIH']
        data_wt =data.loc[data['y_1'] == 0, 'MSIH']
        stats , p_value = stats.ttest_ind(data_mut, data_wt)
    
    keys = list(target_labelDict.keys())
    
    for key in keys:
        y_pred = resultsPath[key]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label = target_labelDict[key])
        print('TOTAL AUC FOR target {} IN THIS DATASET IS : {} '.format(key, np.round(metrics.auc(fpr, tpr), 3)))
        auc_values = []
        nsamples = 1000
        rng = np.random.RandomState(666)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        for i in range(nsamples):
            indices = rng.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_pred[indices])) < 2 or np.sum(y_true[indices]) == 0:
                continue    
            fpr, tpr, thresholds = metrics.roc_curve(y_true[indices], y_pred[indices], pos_label = target_labelDict[key])
            auc_values.append(metrics.auc(fpr, tpr))
        
        auc_values = np.array(auc_values)
        auc_values.sort()
        try :
            low_value = np.round(auc_values[int(0.025 * len(auc_values))], 3)
            high_value = np.round(auc_values[int(0.975 * len(auc_values))], 3)
        except IndexError :
            low_value = auc_values
            high_value = auc_values       
    
    y_pred = list(data['nonMSIH'])
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    fig, ax = plt.subplots()
    aur_value = roc_auc_score(y_true, y_pred)
    plt.title(str(title), fontsize = 15)
    plt.plot(fpr, tpr, linewidth = 3, color='blue', label='ROC curve (area = %0.4f)' % aur_value+'(%0.2f' %low_value +'- %0.2f)' %high_value)
    plt.plot(lw=2, label='ROC curve (area = %0.4f)' % aur_value)
    plt.plot([0, 1], ls = "--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c = ".7")
    plt.ylabel('True Positive Rate', fontsize = 20)
    plt.xlabel('False Positive Rate', fontsize = 20)
    plt.legend(loc="lower right")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.savefig(full_path+savepath+'.svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return  aur_value,low_value,high_value,p_value