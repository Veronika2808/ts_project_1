import optuna
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import pandas as pd
import pickle


def fit_model(X_train, X_val, y_train, y_val):
    """
    Fit Ridge regression with hyperparams tuning by Optuna
    In: X_train, X_val, y_train, y_val
    Out: fitted model with opt hyperparams
    """

    def objective_lr(trial):
    
        param = {
            "alpha": trial.suggest_float("alpha", 0.01, 1),
            "tol": trial.suggest_float("tol", 1e-4, 1e-1),
            "solver": trial.suggest_categorical(
                "solver", ["auto", 'svd', 'cholesky', 'sparse_cg', 'lsqr', 'sag']
            ),
            "random_state": 28
            
        }

        lr = Ridge(**param)

        lr.fit(X_train, y_train)

        preds = lr.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_lr, n_trials=100, timeout=600)
    params= study.best_params

    lr = Ridge(**params)

    lr.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    with open('fitted_model.pkl', 'wb') as f:
        pickle.dump(lr, f)
    
    return lr
