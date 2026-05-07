import pandas
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.regression.linear_model import OLS
import statsmodels.formula.api as smf

from statsmodels.gam.api import GLMGam, BSplines

def gen_true_function(X, complexity=6, pseudo_rng=np.random.default_rng()):
    Bx = BSplines(X, complexity, degree=3)
    beta = pseudo_rng.normal(size=complexity - 1)
    def ff(x):
        y = Bx.transform(x) @ beta
        return y
    return ff


def gen_observations(n_obs, x, f_x,noise_std=0.1,
                     pseudo_rng=np.random.default_rng()):
    N = len(x)
    obs_int = pseudo_rng.choice(np.arange(N), n_obs, replace=False)
    X_obs = x[obs_int]
    Y_obs = f_x[obs_int] + pseudo_rng.normal(0,noise_std,n_obs)
    return X_obs, Y_obs



def fit_poly(x, y, order=0):
    bx = (np.
          polynomial.legendre.legvander(x, order))
    mod = OLS(y, bx).fit()
    return mod


def polypredict(x, mod, order=0):
    bx = (np.polynomial.legendre.legvander(x, order))
    Y_hat = mod.predict(bx)
    return Y_hat


def do_experiment(complexity, n_obs, X, f, sd=0.1,
                  pseudo_rng=np.random.default_rng()):
    true_Y = f(X)
    x_dash,y_dash =gen_observations(n_obs,X,true_Y,sd,pseudo_rng)
    mod = fit_poly(x_dash,y_dash,order=complexity)
    y_hat = polypredict(x_dash,mod,order=complexity)
    f_hat = polypredict(X,mod,order=complexity)

    Y_error = y_dash - y_hat
    SS_error = np.sum(Y_error**2)

    out = {
        "SSq_error":SS_error,
        "complexity":complexity,
        "n_obs":n_obs,
        "error_sd":sd
    }
    return {"summary":out, "f_hat":f_hat}