"""
Generates a PyMC3 model: the modified Rachlin model _without_ any hierarchican inference
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import math


def parameter_estimation(data, sample_options):
    model = generate_model(data)
    with model:
        trace = pm.sample(**sample_options)
    return extract_info_from_trace(trace)


def generate_model(data):
    """Generate a PyMC3 model with the given observed data"""

    # decant data
    R = data["R"].values
    RA, DA = data["RA"].values, data["DA"].values
    RB, DB = data["RB"].values, data["DB"].values

    with pm.Model() as model:
        # define priors
        logk = pm.Normal("logk", mu=-3.760, sd=3)
        logs = pm.Normal("logs", mu=0, sd=1)

        VA = pm.Deterministic("VA", value_function(RA, DA, logk, logs))
        VB = pm.Deterministic("VB", value_function(RB, DB, logk, logs))
        P_chooseB = pm.Deterministic("P_chooseB", choice_psychometric(VB - VA))

        R = pm.Bernoulli("R", p=P_chooseB, observed=R)

    return model


# helper functions for the model


def value_function(reward, delay, logk, logs):
    """Calculate the present subjective value of a given prospect"""
    k = pm.math.exp(logk)
    s = pm.math.exp(logs)
    return reward / (1.0 + (k * delay) ** s)


def choice_psychometric(x, ϵ=0.01):
    # x is the decision variable
    return ϵ + (1.0 - 2.0 * ϵ) * (1 / (1 + pm.math.exp(-1.7 * (x))))


def trace_quantiles(x):
    return pd.DataFrame(pm.quantiles(x, [2.5, 5, 50, 95, 97.5]))


def extract_info_from_trace(trace):
    """Return a 1-row DataFrame of summary statistics (i.e. means, ranges)
    of the parameters of interest"""

    # useful PyMC3 function to get summary statistics
    summary = pm.summary(trace, ["logk", "logs"])

    logk = summary["mean"]["logk"]
    logkL = summary["hpd_2.5"]["logk"]
    logkU = summary["hpd_97.5"]["logk"]

    logs = summary["mean"]["logs"]
    logsL = summary["hpd_2.5"]["logs"]
    logsU = summary["hpd_97.5"]["logs"]

    return pd.DataFrame(
        {
            "logk_est": [logk],
            "logk_est_L": [logkL],
            "logk_est_U": [logkU],
            "logs_est": [logs],
            "logs_est_L": [logsL],
            "logs_est_U": [logsU],
        }
    )
