from __future__ import division

import igraph
import numpy as np
import powerlaw
from scipy.stats import binom

# no transformation
def no_transform(feature, **kwargs):
    return np.array(feature)

# tranform feature to quantile
def quantile_transform(feature, **kwargs):
    total = len(feature)
    feature = np.array(feature)
    # feature - quantile mapping
    D = {}
    for f in np.unique(feature):
        D[f] = len(feature[feature < f]) / total

    quantile = [D[f] for f in feature]
    return np.array(quantile)

# divide-by-average transformation
def average_transform(degree, **kwargs):
    return np.array(degree) / np.mean(degree)

# degree transformation
def degree_transform(degree, **kwargs):
    # pre-processing
    degree = np.array(degree)

    # fitting power-law distribution
    fit = powerlaw.Fit(degree, discrete=True, xmin=(1,6))

    alpha = fit.alpha
    x_min = fit.xmin

    n = len(degree)
    total = len(degree[degree >= x_min])
    c = (alpha - 1) * total / n

    T = {}
    for d in np.unique(degree):
        if (d <= x_min):
            T[d] = d
        else:
            T[d] = np.power(d/x_min, alpha-1) * x_min

    degree = np.round([ T[d] for d in degree ])
    return degree

# degree transformation with fallback
def degree_transform_with_fallback(degree, **kwargs):
    # pre-processing
    degree = np.array(degree)
    total = len(degree[degree < 0])

    # fitting power-law distribution
    fit = powerlaw.Fit(degree, discrete=True)

    alpha = fit.alpha
    sigma = fit.sigma
    x_min = min(6, fit.xmin)

    P = {}; D = {}; T = {}
    for d in np.unique(degree):
        P[d] = len(degree[degree >= d]) / total
        D[d] = d if d <= 1 else 1/P[d]

    # fallback
    if (sigma > 0.05):
        print 'sigma =', sigma, ', fallback!'
        return degree

    c = (alpha - 1) * total / len(degree)

    for d in np.unique(degree):
        if (d <= 1):
            T[d] = d
        else:
            P_r = len(degree[degree == d]) / total
            P_p = np.power(d, -alpha) * c
            T_d = np.power(d, alpha-1)

            if (sigma > 0.05):
                T[d] = (d*(P_r-P_p) + D[d]*P_p) / P_r if d < x_min else D[d]
            else:
                T[d] = (d*(P_r-P_p) + c/d) / P_r if d < x_min else T_d

    degree = np.array([ T[d] for d in degree ])
    return degree

# transform local clustering coeffient
def lcc_transform(lcc, degree):
    degree, lcc = np.array(degree), np.array(lcc)

    s = (degree * (degree - 1) / 2).astype(np.int)
    t = np.round(lcc * s).astype(np.int)
    if sum(s) == 0:
        return lcc

    P = {}
    for S in np.unique(s):
        t_s = t[s == S]
        p0 = len(t_s[t_s == 0]) / len(t_s)
        for T in np.unique(t_s):
            P[(T,S)] = (len(t_s[t_s <= T]) / len(t_s) - p0) / (1 - p0) if p0 < 1 else 0

    lcc = np.array([ P[(t[i], s[i])] for i in range(len(degree)) ])
    return lcc

