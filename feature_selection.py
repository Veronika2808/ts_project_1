def correlation_between_features_and_target(df, corr_thr=0.5):
    """
    Select features by their correlation with target
    In: 
        df: intime dataset with generated features
        corr_thr: correlation threshold to filter features
    Out:
        list of selected features

    """
    X = df.drop(columns=['target'])
    y = df['target']
    cols_to_drop = []
    for col in X.columns:
        if X[col].corr(y) <= corr_thr:
            cols_to_drop.append(col)
    X_new = X.drop(cols_to_drop, axis=1)
    
    return X_new.columns




