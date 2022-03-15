# Filename: fit.py
# -*- coding: utf-8 -*-
# pylint: disable=locally-disabled
"""
A collection of fit functions and modules.

"""
import numpy as np
import km3pipe.extras


def fit_delta_ts(data, time_s):
    """Fits gaussians to delta t for each PMT pair.

    Parameters
    ----------
    data: 2d np.array: x = PMT combinations (465), y = time, entry = frequency
    time_s: length of data taking in seconds

    Returns
    -------
    numpy arrays with rates and means for all PMT combinations
    """
    scipy = km3pipe.extras.scipy()
    from scipy import optimize

    data = data / time_s

    xs = np.arange(-20, 21)

    def gaussian(x, mean, sigma, rate, offset):
        return (
            rate / np.sqrt(2 * np.pi) / sigma * np.exp(-((x - mean) ** 2) / sigma ** 2)
            + offset
        )

    rates = []
    means = []
    for combination in data:
        try:
            popt, _ = optimize.curve_fit(gaussian, xs, combination, p0=[0, 2, 1000, 20])
        except RuntimeError:
            popt = (0, 0, 0, 0)
        rates.append(popt[2])
        means.append(popt[0])
    return np.array(rates), np.array(means)
