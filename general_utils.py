import pandas as pd
import numpy as np

def categorize_objects(sample_df):
    live_samp = sample_df.copy()
    
    for column in live_samp.columns:
        if live_samp[column].dtype == np.object:
            live_samp[column] = live_samp[column].astype('category')
            
    return live_samp

def dummify_all_categories(sample_df, target):
    live_samp = sample_df.copy()

    cat_dummies = []
    drop_cols = []
    for column in live_samp.columns:
        if column != target and hasattr(live_samp[column], 'cat'):
            live_samp[column] = live_samp[column].cat.add_categories(['UNK'])
            live_samp[column].fillna('UNK')
            hot = pd.get_dummies(live_samp[column], prefix=column)
            cat_dummies.append(hot)
            drop_cols.append(column)

    live_samp = live_samp.drop(drop_cols, axis=1)            

    one_hot_enc = pd.concat(cat_dummies, axis=1)

    model_samp = pd.concat([live_samp, one_hot_enc], axis=1)
    feat_cols = model_samp.columns.drop([target])
        
    return model_samp, feat_cols

def standardize_series_value(series):
    live_samp = series.copy()
    
    live_samp = (live_samp - live_samp.mean()) / live_samp.std()
    
    return live_samp