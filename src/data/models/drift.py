import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(train_df, new_df):

    report = {}

    for col in train_df.columns:

        stat, pvalue = ks_2samp(train_df[col], new_df[col])

        report[col] = pvalue

    return report