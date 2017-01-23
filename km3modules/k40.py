# coding=utf-8
# Filename: k40.py
# pylint: disable=locally-disabled
"""
A collection of k40 related functions and modules.

"""
from __future__ import division, absolute_import, print_function

import os
from itertools import combinations

from scipy import optimize
import numpy as np


import km3pipe as kp

__author__ = "Jonas Reubelt"
__email__ = "jreubelt@km3net.de"
__status__ = "Development"

log = kp.logger.logging.getLogger(__name__)  # pylint: disable=C0103
# log.setLevel(logging.DEBUG)





def calibrate_t0s(filename, dom_id, detx):
    """Calibrate intra DOM PMT time offsets"""
    loaders = {'.txt': load_k40_coincidences_from_txt,
               '.root': load_k40_coincidences_from_rootfile}
    try:
        loader = loaders[os.path.splitext(filename)[1]]
    except KeyError:
        log.critical('File format not supported.')
        raise IOError
    else:
        data, weight = loader(filename, dom_id)
    rates, means = fit_delta_ts(data, weight)
    angles = calculate_angles(detx)
    fitted_rates, _ = fit_angular_distribution(angles, rates)
    opt_t0s = minimize_t0s(means, fitted_rates)
    return opt_t0s

def load_k40_coincidences_from_txt(filename, dom_id):
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
    for c in jmonitork40_comb_indices:
        combination = []
        for b in range(1, histo_2d_monitor.GetNbinsY() + 1):
            combination.append(histo_2d_monitor.GetBinContent(c, b))
        data.append(combination)
    
    weights = {}
    weights_histo = root_file_monitor.Get('weights_hist')
    for i in range(1, weights_histo.GetNbinsX() + 1):
        weight = weights_histo.GetBinContent(i)
        label = weights_histo.GetXaxis().GetBinLabel(i)
        weights[label[3:]] = weight
    try:
        data = np.array(data) / weights[dom_id]
    except KeyError:
        log.critical('DOM Id {0} not found.'.format(dom_id))
        raise KeyError
    return np.array(data), weights[dom_id]


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
    start = -(data.shape[1] - 1) / 2
    end = -start + 1
    xs = np.arange(start, end)
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


def minimize_t0s(means, weights):
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
    def make_quality_function(means, weights):
        combs = list(combinations(range(31), 2))
        def quality_function(t0s):
            sq_sum = 0
            for mean, comb, weight in zip(means, combs, weights):
                sq_sum += ((mean - (t0s[comb[1]] - t0s[comb[0]])) * weight)**2
            return sq_sum
        return quality_function

    qfunc = make_quality_function(means, weights)
    #t0s = np.zeros(31)
    t0s = np.random.rand(31)
    bounds = [(0, 0)]+[(-10., 10.)] * 30
    opt_t0s = optimize.minimize(qfunc, t0s, bounds=bounds)
    return opt_t0s


def minimize_qes(fitted_rates, rates, weights):
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
    corrected_means = np.array([(opt_t0s[comb[1]] - opt_t0s[comb[0]]) 
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
    corrected_rates = np.array([rate / opt_qes[comb[0]] / opt_qes[comb[1]]  \
                            for rate, comb in zip(rates, combs)])
    return corrected_rates


jmonitork40_comb_indices =  \
    np.array((254, 423, 424, 391, 392, 255, 204, 205, 126, 120, 121, 0,
    22, 12, 80, 81, 23, 48, 49, 148, 150, 96, 296, 221, 190, 191, 297, 312,
    313, 386, 355, 132, 110, 431, 42, 433, 113, 256, 134, 358, 192, 74,
    176, 36, 402, 301, 270, 69, 384, 2, 156, 38, 178, 70, 273, 404, 302,
    77, 202, 351, 246, 440, 133, 262, 103, 118, 44, 141, 34, 4, 64, 30,
    196, 91, 172, 61, 292, 84, 157, 198, 276, 182, 281, 410, 381, 289,
    405, 439, 247, 356, 102, 263, 119, 140, 45, 35, 88, 65, 194, 31,
    7, 60, 173, 82, 294, 158, 409, 277, 280, 183, 200, 288, 382, 406,
    212, 432, 128, 388, 206, 264, 105, 72, 144, 52, 283, 6, 19, 14,
    169, 24, 310, 97, 379, 186, 218, 59, 93, 152, 317, 304, 111, 387,
    129, 207, 104, 265, 73, 18, 53, 5, 284, 146, 168, 15, 308, 26,
    98, 92, 187, 58, 219, 380, 316, 154, 305, 112, 434, 257, 357, 135,
    193, 300, 177, 401, 37, 75, 68, 271, 1, 385, 159, 403, 179, 272,
    71, 39, 76, 303, 203, 213, 393, 248, 442, 298, 145, 184, 89, 377,
    315, 216, 57, 309, 27, 99, 8, 54, 16, 171, 287, 153, 21, 78,
    394, 441, 249, 299, 314, 185, 376, 90, 147, 56, 217, 25, 311, 100,
    286, 55, 170, 17, 9, 20, 155, 79, 425, 426, 383, 306, 220, 290,
    291, 307, 188, 189, 149, 151, 101, 86, 13, 50, 51, 87, 28, 29,
    3, 352, 399, 375, 274, 407, 197, 285, 180, 279, 83, 295, 160, 199,
    66, 174, 63, 33, 10, 95, 40, 400, 282, 275, 195, 408, 378, 278,
    181, 293, 85, 161, 32, 67, 62, 175, 201, 94, 11, 41, 435, 415,
    359, 360, 436, 347, 348, 258, 259, 318, 136, 162, 222, 223, 137, 114,
    115, 43, 451, 443, 266, 389, 335, 456, 208, 396, 363, 250, 238, 327,
    235, 107, 130, 215, 116, 343, 344, 452, 461, 462, 331, 332, 417, 226,
    324, 371, 372, 229, 240, 241, 163, 142, 267, 230, 412, 122, 428, 319,
    353, 227, 340, 166, 47, 108, 253, 138, 444, 411, 231, 427, 123, 320,
    46, 228, 165, 341, 354, 252, 109, 139, 455, 336, 395, 209, 364, 106,
    239, 234, 328, 251, 214, 131, 117, 373, 447, 243, 418, 164, 369, 325,
    460, 342, 329, 237, 224, 242, 448, 419, 339, 370, 459, 326, 167, 236,
    330, 225, 127, 365, 124, 333, 244, 450, 430, 397, 211, 260, 366, 429,
    334, 449, 245, 125, 210, 398, 261, 321, 420, 421, 422, 322, 367, 368,
    323, 345, 413, 232, 143, 268, 446, 361, 463, 464, 346, 453, 454, 416,
    374, 233, 337, 458, 349, 414, 457, 338, 350, 445, 269, 362, 390, 437, 438))

