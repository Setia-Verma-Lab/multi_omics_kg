import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import average_precision_score

all_prs = 'all_applied_prs_scores.csv'
samples_keep = 'samples.keep.txt'
samples_split = 'patient_predictions.csv'
samples_covariates = 'patient_covariates.csv'
samples_labels = 'chart_review_and_EHR_labels.csv'


all_prs_df = pd.read_csv(all_prs, usecols = ['IID', 'GRS_ALL_No_PMBB_W_SCORE_SUM'])

samples_keep_df = pd.read_csv(samples_keep)
samples_keep_df.columns = ['PMBB_ID']

samples_split_df = pd.read_csv(samples_split, usecols=['PMBB_ID', 'status'])

# merge all_prs_df onto samples_keep
merged_df = pd.merge(samples_keep_df, all_prs_df, left_on='PMBB_ID', right_on='IID', how='left')

# merge covariates onto the merged dataframe above
samples_covariates_df = pd.read_csv(samples_covariates, usecols = ['PMBB_ID', 'Sample_age', 'imputed_PC1', 'imputed_PC2', 'imputed_PC3', 'imputed_PC4', 'imputed_PC5'])

relevant_samples_df = pd.merge(merged_df, samples_covariates_df, on='PMBB_ID', how='left')

labels_df = pd.read_csv(samples_labels, usecols=['PMBB_ID', 'Chart_Adeno_or_Endo'])
relevant_samples_labels_df = pd.merge(relevant_samples_df, labels_df, on='PMBB_ID', how='left')
relevant_samples_labels_df.dropna(inplace=True) # drops 39 rows with missing PC values

all_data = pd.merge(relevant_samples_labels_df, samples_split_df, on='PMBB_ID', how='left')

# --------------------------------- 

# train / val split
train_df = all_data[all_data['status']=='train'].copy()
val_df   = all_data[all_data['status']=='val'].copy()

print(f"Train: {len(train_df)} subjects,  Val: {len(val_df)} subjects\n")

# get predictive performance
clf = LogisticRegression(solver='liblinear')

features = ['GRS_ALL_No_PMBB_W_SCORE_SUM'] # no age and ancestry
X_train = train_df[features]
y_train = train_df['Chart_Adeno_or_Endo']
X_val   = val_df[features]
y_val   = val_df['Chart_Adeno_or_Endo']

clf.fit(X_train, y_train)

def bootstrap_metrics(y_true, y_prob, y_pred, n_boot=1000, seed=42):
    """
    Returns dict:
      {
        'auc':  (mean, std),
        'acc':  (mean, std),
        'f1':   (mean, std),
        'auprc':(mean, std)
      }
    computed by bootstrapping the array indices.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = {'auc':[], 'acc':[], 'f1':[], 'auprc':[]}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)        # sample with replacement
        yt = y_true[idx]
        yp = y_prob[idx]
        yd = y_pred[idx]

        stats['auc'].append(   roc_auc_score(yt, yp) )
        stats['acc'].append(   accuracy_score(yt, yd) )
        stats['f1'].append(    f1_score(yt, yd) )
        stats['auprc'].append( average_precision_score(yt, yp) )

    # convert to arrays, compute mean & std
    return {k: (np.mean(v), np.std(v, ddof=1)) for k,v in stats.items()}

def evaluate_with_bootstrap(name, X, y, clf, n_boot=1000):
    # get probabilities and hard preds
    y_prob = clf.predict_proba(X)[:, 1]
    # y_pred = clf.predict(X) # uses standard 0.5 as threshold
    y_pred = (clf.predict_proba(X)[:,1] >= 0.5).astype(bool)
    
    y_true = y.values if hasattr(y, 'values') else np.asarray(y)

    # point‐estimates
    auc0   = roc_auc_score(y_true, y_prob)
    acc0   = accuracy_score(y_true, y_pred)
    f10    = f1_score(y_true, y_pred)
    auprc0 = average_precision_score(y_true, y_prob)

    # bootstrap for std
    boot = bootstrap_metrics(y_true, y_prob, y_pred, n_boot=n_boot)

    print(f"{name}:")
    print(f"roc auc:  {auc0:.3f} ± {boot['auc'][1]:.3f}")
    print(f"accuracy: {acc0:.3f} ± {boot['acc'][1]:.3f}")
    print(f"f1: {f10:.3f} ± {boot['f1'][1]:.3f}")
    print(f"auprc:    {auprc0:.3f} ± {boot['auprc'][1]:.3f}")

# evaluate_with_bootstrap("TRAIN set", X_train, y_train, clf)
evaluate_with_bootstrap("val set", X_val, y_val, clf)