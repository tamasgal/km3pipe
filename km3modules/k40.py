# coding=utf-8
# Filename: k40.py
# pylint: disable=locally-disabled
"""
A collection of k40 related functions and modules.

"""
from __future__ import division, absolute_import, print_function

from itertools import combinations

from scipy import optimize
import numpy as np


import km3pipe as kp

__author__ = "Jonas Reubelt"
__email__ = "jreubelt@km3net.de"
__status__ = "Development"

log = kp.logger.logging.getLogger(__name__)  # pylint: disable=C0103
# log.setLevel(logging.DEBUG)


def load_k40_coincidences_from_txt(filename):
    """Load k40 coincidences from txt file"""
    data = np.loadtxt(filename)
    return data


def load_k40_coincidences_from_rootfile(filename, dom_id):
    """Load k40 coincidences from JMonitorK40 ROOT file"""
    from ROOT import TFile
    root_file_monitor = TFile( filename, "READ" )
    dom_name = dom_id + ".2S"
    histo_2d_monitor = root_file_monitor.Get(dom_name)
    data = []
    for i in range(1, histo_2d_monitor.GetNbinsX() + 1):
        combination = []
        for j in range(1, histo_2d_monitor.GetNbinsY() + 1):
            combination.append(histo_2d_monitor.GetBinContent(i, j))
        data.append(combination)
    return np.array(data)


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

    data = data / time_s

    xs = np.arange(-20, 21)
    def gaussian(x, mean, sigma, rate, offset):
        return rate / np.sqrt(2 * np.pi) /  \
               sigma * np.exp(-(x-mean)**2 / sigma**2) + offset
    rates = []
    means = []
    for combination in data:
        try:
            popt, _ = optimize.curve_fit(gaussian, xs, combination,
                                         p0=[0, 2, 1000, 20])
        except RuntimeError:
            popt = (0, 0, 0, 0)
        rates.append(popt[2])
        means.append(popt[0])
    return np.array(rates), np.array(means)


def calculate_angles(detector_file):
    """Calculates angles between PMT combinations according to positions in 
    detector_file
    
    Parameters
    ----------
    detector_file: file from which to read the PMT positions (.detx)
    
    Returns
    -------
    angles: numpy array of angles between all PMT combinations

    """    
    angles = []
    pmt_combinations = list(combinations(range(31), 2))
    det = kp.hardware.Detector(filename = detector_file)
    pmt_angles = det.pmt_angles
    for first, second in pmt_combinations:
        angles.append(kp.tools.angle_between(np.array(pmt_angles[first]), 
                                             np.array(pmt_angles[second])))
    return np.array(angles)


def exponential_polinomial(x, p1, p2, p3, p4):
    return  1 * np.exp(p1 + x * (p2 + x * (p3 + x * p4)))


def fit_angular_distribution(angles, rates):
    """Fits angular distribution of rates. 
        
    Parameters
    ----------
    rates: numpy array with rates for all PMT combinations
    angles: numpy array with angles for all PMT combinations
    
    Returns
    -------
    fitted_rates: numpy array of fitted rates (fit_function(angles, *popt))

    """    
    cos_angles = np.cos(angles)
    popt, _ = optimize.curve_fit(exponential_polinomial,cos_angles, rates)
    fitted_rates = exponential_polinomial(cos_angles, *popt)
    return fitted_rates, popt


def minimize_t0s(fitted_rates, means, weights=np.ones(465)):
    """Varies t0s to minimize the deviation of the gaussian means from zero.
    
    Parameters
    ----------
    fitted_rates: numpy array of fitted rates from fit_angular_distribution 
    means: numpy array of means of all PMT combinations
    weights: numpy array of weights for the squared sum 

    Returns
    -------
    opt_t0s: optimal t0 values for all PMTs

    """
    def make_quality_function(fitted_rates, means, weights):
        combs = list(combinations(range(31), 2))
        def quality_function(t0s):
            sq_sum = 0
            for fitted_rate, mean, comb, weight  \
                    in zip(fitted_rates, means, combs, weights):
                sq_sum += ((mean - (t0s[comb[1]] - t0s[comb[0]])) * weight)**2
            return sq_sum
        return quality_function

    qfunc = make_quality_function(fitted_rates, means, weights)
    t0s = np.zeros(31)
    bounds = [(0, 0)]+[(-10., 10.)] * 30
    opt_t0s = optimize.minimize(qfunc, t0s, bounds=bounds)
    return opt_t0s


def minimize_qes(fitted_rates, rates, weights)):
    """Varies QEs to minimize the deviation of the rates from the fitted_rates.
    
    Parameters
    ----------
    fitted_rates: numpy array of fitted rates from fit_angular_distribution 
    rates: numpy array of rates of all PMT combinations
    weights: numpy array of weights for the squared sum   
    
    Returns
    -------
    opt_qes: optimal qe values for all PMTs

    """
    def make_quality_function(fitted_rates, rates, weights):
        combs = list(combinations(range(31), 2))
        def quality_function(qes):
            sq_sum = 0
            for fitted_rate, comb, rate, weight  \
                    in zip(fitted_rates, combs, rates, weights):
                sq_sum += ((rate / qes[comb[0]] / qes[comb[1]] 
                            - fitted_rate) * weight)**2
            return sq_sum
        return quality_function

    qfunc = make_quality_function(fitted_rates, rates, weights)
    qes = np.ones(31)
    bounds = [(0.1, 2.)] * 31
    opt_qes = optimize.minimize(qfunc, qes, bounds=bounds)
    return opt_qes


def correct_means(means, opt_t0s):
    """Applies optimal t0s to gaussians means. 
    
    Should be around zero afterwards.
    
    Parameters
    ----------
    means: numpy array of means of gaussians of all PMT combinations
    opt_t0s: numpy array of optimal t0 values for all PMTs
    
    Returns
    -------
    corrected_means: numpy array of corrected gaussian means for all PMT combs

    """
    combs = list(combinations(range(31), 2))
    corrected_means = np.array([(opt_t0s[combs[1]] - opt_t0s[combs[0]]) 
                                - mean for mean, comb in zip(means, combs)])
    return corrected_means


def correct_rates(rates, opt_qes):
    """Applies optimal qes to rates. 

    Should be closer to fitted_rates afterwards.
    
    Parameters
    ----------
    rates: numpy array of rates of all PMT combinations
    opt_qes: numpy array of optimal qe values for all PMTs
    
    Returns
    -------
    corrected_rates: numpy array of corrected rates for all PMT combinations
    """
    
    combs = list(combinations(range(31), 2))
    corrected_rates = np.array([rate / opt_qes[comb[0] / opt_qes[comb[1]]  \
                            for rate, comb in zip(rates, combs)])
    return corrected_rates
