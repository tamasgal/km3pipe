# Filename: k40.py
# -*- coding: utf-8 -*-
# pylint: disable=locally-disabled
"""
A collection of k40 related functions and modules.

"""

import os
from itertools import combinations
from collections import defaultdict
from functools import partial
from datetime import datetime
import io

import numpy as np
import pickle

import km3pipe as kp
import km3pipe.extras
from km3pipe.io.daq import TMCHData

log = kp.logger.get_logger(__name__)  # pylint: disable=C0103
try:
    from numba import jit
except ImportError:
    log.warning(
        "This module requires `numba` to be installed, otherwise "
        "the functions and Modules imported from this module can "
        "be painfully slow."
    )
    jit = lambda f: f

__author__ = "Jonas Reubelt"
__email__ = "jreubelt@km3net.de"
__status__ = "Development"

# log.setLevel(logging.DEBUG)

TIMESLICE_LENGTH = 0.1  # [s]
MC_ANG_DIST = np.array([-0.72337394, 2.59196335, -0.43594182, 1.10514914])


class K40BackgroundSubtractor(kp.Module):
    """Subtracts random coincidence background from K40 data

    Notes
    -----
    Requires servce 'MedianPMTRates()'
    Writes 'K40Counts' into the Blob: dict, Corrected K40 counts

    """

    def configure(self):
        self.combs = list(combinations(range(31), 2))
        self.mode = self.get("mode", default="online")
        self.expose(self.get_corrected_counts, "GetCorrectedTwofoldCounts")
        self.corrected_counts = None

    def process(self, blob):
        if self.mode != "online":
            return blob
        self.cprint("Subtracting random background calculated from single rates")
        corrected_counts = self.subtract_background()
        blob["CorrectedTwofoldCounts"] = corrected_counts

        return blob

    def get_corrected_counts(self):
        return self.corrected_counts

    def subtract_background(self):
        counts = self.services["TwofoldCounts"]
        dom_ids = list(counts.keys())
        mean_rates = self.services["GetMedianPMTRates"]()
        corrected_counts = {}
        livetimes = self.services["GetLivetime"]()
        for dom_id in dom_ids:
            try:
                pmt_rates = mean_rates[dom_id]
            except KeyError:
                log.warning("Skipping BG correction for DOM {}.".format(dom_id))
                corrected_counts[dom_id] = counts[dom_id]
                continue
            livetime = livetimes[dom_id]
            k40_rates = counts[dom_id] / livetime
            bg_rates = []
            for c in self.combs:
                bg_rates.append(pmt_rates[c[0]] * pmt_rates[c[1]] * 1e-9)
            corrected_counts[dom_id] = (k40_rates.T - np.array(bg_rates)).T * livetime
        return corrected_counts

    def finish(self):
        if self.mode == "offline":
            self.cprint("Subtracting background calculated from summaryslices.")
            self.corrected_counts = self.subtract_background()

    def dump(self, mean_rates, corrected_counts, livetime):
        pickle.dump(mean_rates, open("mean_rates.p", "wb"))
        pickle.dump(
            {"data": corrected_counts, "livetime": livetime},
            open("k40_counts_bg_sub.p", "wb"),
        )


class IntraDOMCalibrator(kp.Module):
    """Intra DOM calibrator which performs the calibration from K40Counts.

    Parameters
    ----------
    det_id: int
      Detector ID [default: 14]
    ctmin: float
      Minimum cos(angle)
    mode: str ('offline' | 'online')
      Calibration mode [default: 'online']

    Notes
    -----
    Requires 'TwofoldCounts': dict
        (key=dom_id, value=matrix of k40 counts 465x(dt*2+1))
    Requires 'CorrectedTwofoldCounts': dict
        (key=dom_id, value=matrix of k40 counts 465x(dt*2+1))
    Writes 'IntraDOMCalibration' into the blob: dict (key=dom_id, value=calibration)

    """

    def configure(self):
        det_id = self.get("det_id") or 14
        self.detector = kp.hardware.Detector(det_id=det_id)
        self.ctmin = self.require("ctmin")
        self.mode = self.get("mode", default="online")
        self.calib_filename = self.get("calib_filename", default="k40_cal.p")

    def process(self, blob):
        if self.mode != "online":
            return blob

        if "CorrectedTwofoldCounts" in blob:
            log.info("Using corrected twofold counts")
            fit_background = False
            twofold_counts = blob["CorrectedTwofoldCounts"]
        else:
            log.info("No corrected twofold counts found, fitting background.")
            twofold_counts = self.services["TwofoldCounts"]
            fit_background = True

        blob["IntraDOMCalibration"] = self.calibrate(twofold_counts, fit_background)
        return blob

    def calibrate(self, twofold_counts, fit_background=False):
        self.cprint("Starting calibration:")
        calibration = {}

        for dom_id, data in twofold_counts.items():
            self.cprint(" calibrating DOM '{0}'".format(dom_id))
            try:
                livetime = self.services["GetLivetime"]()[dom_id]
                calib = calibrate_dom(
                    dom_id,
                    data,
                    self.detector,
                    livetime=livetime,
                    fit_background=fit_background,
                    ad_fit_shape="exp",
                    ctmin=self.ctmin,
                )
            except RuntimeError:
                log.error(" skipping DOM '{0}'.".format(dom_id))
            except KeyError:
                log.error(" skipping DOM '{0}', no livetime".format(dom_id))
            else:
                calibration[dom_id] = calib

        return calibration

    def finish(self):
        if self.mode == "offline":
            self.cprint("Starting offline calibration")
            if "GetCorrectedTwofoldCounts" in self.services:
                self.cprint("Using corrected twofold counts")
                twofold_counts = self.services["GetCorrectedTwofoldCounts"]()
                fit_background = False
            else:
                self.cprint("Using uncorrected twofold counts")
                twofold_counts = self.services["TwofoldCounts"]
                fit_background = True
            calibration = self.calibrate(twofold_counts, fit_background=fit_background)
            self.cprint("Dumping calibration to '{}'.".format(self.calib_filename))
            with open(self.calib_filename, "wb") as f:
                pickle.dump(calibration, f)


class TwofoldCounter(kp.Module):
    """Counts twofold coincidences in timeslice hits per PMT combination.

    Parameters
    ----------
    'tmax': int
      time window of twofold coincidences [ns]
    'dump_filename': str
      name for the dump file

    Notes
    -----
    Requires the key 'TSHits': RawHitSeries
    Provides the following services:
    'TwofoldCounts': dict (key=dom_id, value=matrix (465,(dt*2+1)))
    'ResetTwofoldCounts': reset the TwofoldCounts dict
    'GetLivetime()': dict (key=dom_id, value=float)
    'DumpTwofoldCounts': Writes twofold counts into 'dump_filename'

    """

    def configure(self):
        self.tmax = self.get("tmax") or 20
        self.dump_filename = self.get("dump_filename")

        self.counts = None
        self.n_timeslices = None
        self.start_time = datetime.utcnow()
        self.reset()

        self.expose(self.counts, "TwofoldCounts")
        self.expose(self.get_livetime, "GetLivetime")
        self.expose(self.reset, "ResetTwofoldCounts")
        if self.dump_filename is not None:
            self.expose(self.dump, "DumpTwofoldCounts")

        if "GetSkippedFrames" in self.services:
            self.skipped_frames = self.services["GetSkippedFrames"]()
        else:
            self.skipped_frames = None

    def reset(self):
        """Reset coincidence counter"""
        self.counts = defaultdict(partial(np.zeros, (465, self.tmax * 2 + 1)))
        self.n_timeslices = defaultdict(int)

    def get_livetime(self):
        return {dom_id: n * TIMESLICE_LENGTH for dom_id, n in self.n_timeslices.items()}

    def process(self, blob):
        log.debug("Processing timeslice")
        hits = blob["TSHits"]

        frame_index = blob["TimesliceInfo"].frame_index
        dom_ids = set(np.unique(hits.dom_id))
        if self.skipped_frames is not None:
            skipped_dom_ids = set(self.skipped_frames[frame_index])
        else:
            skipped_dom_ids = set()

        for dom_id in dom_ids - skipped_dom_ids:
            self.n_timeslices[dom_id] += 1
            mask = hits.dom_id == dom_id
            times = hits.time[mask]
            channel_ids = hits.channel_id[mask]
            sort_idc = np.argsort(times, kind="quicksort")
            add_to_twofold_matrix(
                times[sort_idc],
                channel_ids[sort_idc],
                self.counts[dom_id],
                tmax=self.tmax,
            )
        return blob

    def dump(self):
        """Write coincidence counts into a Python pickle"""
        self.cprint("Dumping data to {}".format(self.dump_filename))
        pickle.dump(
            {"data": self.counts, "livetime": self.get_livetime()},
            open(self.dump_filename, "wb"),
        )


class HRVFIFOTimesliceFilter(kp.Module):
    """Creat a frame index lookup table which holds DOM IDs of frames with
    at least one PMT in HRV."""

    def configure(self):
        filename = self.require("filename")
        filter_hrv = self.get("filter_hrv", default=False)
        self.expose(self.get_skipped_frames, "GetSkippedFrames")
        self.skipped_frames = defaultdict(list)
        p = kp.io.jpp.SummaryslicePump(filename=filename)
        for b in p:
            sum_info = b["SummarysliceInfo"]
            frame_index = sum_info.frame_index
            summaryslice = b["Summaryslice"]
            for dom_id, sf in summaryslice.items():
                if not sf["fifo_status"] or (filter_hrv and any(sf["hrvs"])):
                    self.skipped_frames[frame_index].append(dom_id)

    def get_skipped_frames(self):
        return self.skipped_frames


class SummaryMedianPMTRateService(kp.Module):
    def configure(self):
        self.expose(self.get_median_rates, "GetMedianPMTRates")
        self.filename = self.require("filename")

    def get_median_rates(self):
        rates = defaultdict(list)

        if "GetSkippedFrames" in self.services:
            skipped_frames = self.services["GetSkippedFrames"]()
        else:
            skipped_frames = None

        p = kp.io.jpp.SummaryslicePump(filename=self.filename)
        for b in p:
            sum_info = b["SummarysliceInfo"]
            frame_index = sum_info.frame_index
            summary = b["Summaryslice"]
            for dom_id in summary.keys():
                if skipped_frames is not None and dom_id in skipped_frames[frame_index]:
                    continue
                rates[dom_id].append(summary[dom_id]["rates"])

        median_rates = {}
        for dom_id in rates.keys():
            median_rates[dom_id] = np.median(rates[dom_id], axis=0)

        return median_rates


class MedianPMTRatesService(kp.Module):
    def configure(self):
        self.rates = defaultdict(lambda: defaultdict(list))
        self.expose(self.get_median_rates, "GetMedianPMTRates")

    def process(self, blob):
        try:
            tmch_data = TMCHData(io.BytesIO(blob["CHData"]))
        except ValueError as e:
            self.log.error(e)
            self.log.error("Skipping corrupt monitoring channel packet.")
            return
        dom_id = tmch_data.dom_id
        for channel_id, rate in enumerate(tmch_data.pmt_rates):
            self.rates[dom_id][channel_id].append(rate)

    def get_median_rates(self):
        self.cprint("Calculating median PMT rates.")
        median_rates = {}
        for dom_id in self.rates.keys():
            median_rates[dom_id] = [np.median(self.rates[dom_id][c]) for c in range(31)]
        self.rates = defaultdict(lambda: defaultdict(list))
        return median_rates


class ResetTwofoldCounts(kp.Module):
    def process(self, blob):
        if "DumpTwofoldCounts" in self.services:
            self.cprint("Request twofold dump...")
            self.services["DumpTwofoldCounts"]()
        self.cprint("Resetting twofold counts")
        self.services["ResetTwofoldCounts"]()
        return blob


def calibrate_dom(
    dom_id,
    data,
    detector,
    livetime=None,
    fit_ang_dist=False,
    scale_mc_to_data=True,
    ad_fit_shape="pexp",
    fit_background=True,
    ctmin=-1.0,
):
    """Calibrate intra DOM PMT time offsets, efficiencies and sigmas

    Parameters
    ----------
    dom_id: DOM ID
    data: dict of coincidences or root or hdf5 file
    detector: instance of detector class
    livetime: data-taking duration [s]
    fixed_ang_dist: fixing angular distribution e.g. for data mc comparison
    auto_scale: auto scales the fixed angular distribution to the data

    Returns
    -------
    return_data: dictionary with fit results
    """

    if isinstance(data, str):
        filename = data
        loaders = {
            ".h5": load_k40_coincidences_from_hdf5,
            ".root": load_k40_coincidences_from_rootfile,
        }
        try:
            loader = loaders[os.path.splitext(filename)[1]]
        except KeyError:
            log.critical("File format not supported.")
            raise IOError
        else:
            data, livetime = loader(filename, dom_id)

    combs = np.array(list(combinations(range(31), 2)))
    angles = calculate_angles(detector, combs)
    cos_angles = np.cos(angles)
    angles = angles[cos_angles >= ctmin]
    data = data[cos_angles >= ctmin]
    combs = combs[cos_angles >= ctmin]

    try:
        fit_res = fit_delta_ts(data, livetime, fit_background=fit_background)
        rates, means, sigmas, popts, pcovs = fit_res
    except:
        return 0

    rate_errors = np.array([np.diag(pc)[2] for pc in pcovs])
    # mean_errors = np.array([np.diag(pc)[0] for pc in pcovs])
    scale_factor = None
    if fit_ang_dist:
        fit_res = fit_angular_distribution(
            angles, rates, rate_errors, shape=ad_fit_shape
        )
        fitted_rates, exp_popts, exp_pcov = fit_res
    else:
        mc_fitted_rates = exponential_polinomial(np.cos(angles), *MC_ANG_DIST)
        if scale_mc_to_data:
            scale_factor = np.mean(rates[angles < 1.5]) / np.mean(
                mc_fitted_rates[angles < 1.5]
            )
        else:
            scale_factor = 1.0
        fitted_rates = mc_fitted_rates * scale_factor
        exp_popts = []
        exp_pcov = []
        print("Using angular distribution from Monte Carlo")

    # t0_weights = np.array([0. if a>1. else 1. for a in angles])

    if not fit_background:
        minimize_weights = calculate_weights(fitted_rates, data)
    else:
        minimize_weights = fitted_rates

    opt_t0s = minimize_t0s(means, minimize_weights, combs)
    opt_sigmas = minimize_sigmas(sigmas, minimize_weights, combs)
    opt_qes = minimize_qes(fitted_rates, rates, minimize_weights, combs)
    corrected_means = correct_means(means, opt_t0s.x, combs)
    corrected_rates = correct_rates(rates, opt_qes.x, combs)
    rms_means, rms_corrected_means = calculate_rms_means(means, corrected_means)
    rms_rates, rms_corrected_rates = calculate_rms_rates(
        rates, fitted_rates, corrected_rates
    )
    cos_angles = np.cos(angles)
    return_data = {
        "opt_t0s": opt_t0s,
        "opt_qes": opt_qes,
        "data": data,
        "means": means,
        "rates": rates,
        "fitted_rates": fitted_rates,
        "angles": angles,
        "corrected_means": corrected_means,
        "corrected_rates": corrected_rates,
        "rms_means": rms_means,
        "rms_corrected_means": rms_corrected_means,
        "rms_rates": rms_rates,
        "rms_corrected_rates": rms_corrected_rates,
        "gaussian_popts": popts,
        "livetime": livetime,
        "exp_popts": exp_popts,
        "exp_pcov": exp_pcov,
        "scale_factor": scale_factor,
        "opt_sigmas": opt_sigmas,
        "sigmas": sigmas,
        "combs": combs,
    }
    return return_data


def calculate_weights(fitted_rates, data):
    comb_mean_rates = np.mean(data, axis=1)
    greater_zero = np.array(comb_mean_rates > 0, dtype=int)
    return fitted_rates * greater_zero


def load_k40_coincidences_from_hdf5(filename, dom_id):
    """Load k40 coincidences from hdf5 file

    Parameters
    ----------
    filename: filename of hdf5 file
    dom_id: DOM ID

    Returns
    -------
    data: numpy array of coincidences
    livetime: duration of data-taking
    """

    import h5py

    with h5py.File(filename, "r") as h5f:
        data = h5f["/k40counts/{0}".format(dom_id)]
        livetime = data.attrs["livetime"]
        data = np.array(data)

    return data, livetime


def load_k40_coincidences_from_rootfile(filename, dom_id):
    """Load k40 coincidences from JMonitorK40 ROOT file

    Parameters
    ----------
    filename: root file produced by JMonitorK40
    dom_id: DOM ID

    Returns
    -------
    data: numpy array of coincidences
    dom_weight: weight to apply to coincidences to get rate in Hz
    """

    from ROOT import TFile

    root_file_monitor = TFile(filename, "READ")
    dom_name = str(dom_id) + ".2S"
    histo_2d_monitor = root_file_monitor.Get(dom_name)
    data = []
    for c in range(1, histo_2d_monitor.GetNbinsX() + 1):
        combination = []
        for b in range(1, histo_2d_monitor.GetNbinsY() + 1):
            combination.append(histo_2d_monitor.GetBinContent(c, b))
        data.append(combination)

    weights = {}
    weights_histo = root_file_monitor.Get("weights_hist")
    try:
        for i in range(1, weights_histo.GetNbinsX() + 1):
            # we have to read all the entries, unfortunately
            weight = weights_histo.GetBinContent(i)
            label = weights_histo.GetXaxis().GetBinLabel(i)
            weights[label[3:]] = weight
        dom_weight = weights[str(dom_id)]
    except AttributeError:
        log.info("Weights histogram broken or not found, setting weight to 1.")
        dom_weight = 1.0
    return np.array(data), dom_weight


def gaussian(x, mean, sigma, rate, offset):
    return (
        rate / np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * (x - mean) ** 2 / sigma ** 2)
        + offset
    )


def gaussian_wo_offset(x, mean, sigma, rate):
    return (
        rate / np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * (x - mean) ** 2 / sigma ** 2)
    )


def fit_delta_ts(data, livetime, fit_background=True):
    """Fits gaussians to delta t for each PMT pair.

    Parameters
    ----------
    data: 2d np.array: x = PMT combinations (465), y = time, entry = frequency
    livetime: length of data taking in seconds
    fit_background: if True: fits gaussian with offset, else without offset

    Returns
    -------
    numpy arrays with rates and means for all PMT combinations
    """
    scipy = km3pipe.extras.scipy()
    from scipy import optimize

    data = data / livetime
    start = -(data.shape[1] - 1) / 2
    end = -start + 1
    xs = np.arange(start, end)

    rates = []
    sigmas = []
    means = []
    popts = []
    pcovs = []
    for combination in data:
        mean0 = np.argmax(combination) + start
        try:
            if fit_background:
                popt, pcov = optimize.curve_fit(
                    gaussian,
                    xs,
                    combination,
                    p0=[mean0, 4.0, 5.0, 0.1],
                    bounds=([start, 0, 0, 0], [end, 10, 10, 1]),
                )
            else:
                popt, pcov = optimize.curve_fit(
                    gaussian_wo_offset,
                    xs,
                    combination,
                    p0=[mean0, 4.0, 5.0],
                    bounds=([start, 0, 0], [end, 10, 10]),
                )
        except RuntimeError:
            popt = (0, 0, 0, 0)
        rates.append(popt[2])
        means.append(popt[0])
        sigmas.append(popt[1])
        popts.append(popt)
        pcovs.append(pcov)
    return (
        np.array(rates),
        np.array(means),
        np.array(sigmas),
        np.array(popts),
        np.array(pcovs),
    )


def calculate_angles(detector, combs):
    """Calculates angles between PMT combinations according to positions in
    detector_file

    Parameters
    ----------
    detector: detector description (kp.hardware.Detector)
    combs: pmt combinations

    Returns
    -------
    angles: numpy array of angles between all PMT combinations

    """
    angles = []
    pmt_angles = detector.pmt_angles
    for first, second in combs:
        angles.append(
            kp.math.angle_between(
                np.array(pmt_angles[first]), np.array(pmt_angles[second])
            )
        )
    return np.array(angles)


def exponential_polinomial(x, p1, p2, p3, p4):
    return 1 * np.exp(p1 + x * (p2 + x * (p3 + x * p4)))


def exponential(x, a, b):
    return a * np.exp(b * x)


def fit_angular_distribution(angles, rates, rate_errors, shape="pexp"):
    """Fits angular distribution of rates.

    Parameters
    ----------
    rates: numpy array
      with rates for all PMT combinations
    angles: numpy array
      with angles for all PMT combinations
    shape:
      which function to fit; exp for exponential or pexp for
      exponential_polinomial

    Returns
    -------
    fitted_rates: numpy array of fitted rates (fit_function(angles, popt...))

    """
    if shape == "exp":
        fit_function = exponential
        # p0 = [-0.91871169,  2.72224241, -1.19065965,  1.48054122]
    if shape == "pexp":
        fit_function = exponential_polinomial
        # p0 = [0.34921202, 2.8629577]

    cos_angles = np.cos(angles)
    popt, pcov = optimize.curve_fit(fit_function, cos_angles, rates)
    fitted_rates = fit_function(cos_angles, *popt)
    return fitted_rates, popt, pcov


def minimize_t0s(means, weights, combs):
    """Varies t0s to minimize the deviation of the gaussian means from zero.

    Parameters
    ----------
    means: numpy array of means of all PMT combinations
    weights: numpy array of weights for the squared sum
    combs: pmt combinations to use for minimization

    Returns
    -------
    opt_t0s: optimal t0 values for all PMTs

    """

    def make_quality_function(means, weights, combs):
        def quality_function(t0s):
            sq_sum = 0
            for mean, comb, weight in zip(means, combs, weights):
                sq_sum += ((mean - (t0s[comb[1]] - t0s[comb[0]])) * weight) ** 2
            return sq_sum

        return quality_function

    qfunc = make_quality_function(means, weights, combs)
    # t0s = np.zeros(31)
    t0s = np.random.rand(31)
    bounds = [(0, 0)] + [(-10.0, 10.0)] * 30
    opt_t0s = optimize.minimize(qfunc, t0s, bounds=bounds)
    return opt_t0s


def minimize_sigmas(sigmas, weights, combs):
    """Varies sigmas to minimize gaussian sigma12 - sqrt(sigma1² + sigma2²).

    Parameters
    ----------
    sigmas: numpy array of fitted sigmas of gaussians
    weights: numpy array of weights for the squared sum
    combs: pmt combinations to use for minimization

    Returns
    -------
    opt_sigmas: optimal sigma values for all PMTs

    """

    def make_quality_function(sigmas, weights, combs):
        def quality_function(s):
            sq_sum = 0
            for sigma, comb, weight in zip(sigmas, combs, weights):
                sigma_sqsum = np.sqrt(s[comb[1]] ** 2 + s[comb[0]] ** 2)
                sq_sum += ((sigma - sigma_sqsum) * weight) ** 2
            return sq_sum

        return quality_function

    qfunc = make_quality_function(sigmas, weights, combs)
    s = np.ones(31) * 2.5
    # s = np.random.rand(31)
    bounds = [(0.0, 5.0)] * 31
    opt_sigmas = optimize.minimize(qfunc, s, bounds=bounds)
    return opt_sigmas


def minimize_qes(fitted_rates, rates, weights, combs):
    """Varies QEs to minimize the deviation of the rates from the fitted_rates.

    Parameters
    ----------
    fitted_rates: numpy array of fitted rates from fit_angular_distribution
    rates: numpy array of rates of all PMT combinations
    weights: numpy array of weights for the squared sum
    combs: pmt combinations to use for minimization

    Returns
    -------
    opt_qes: optimal qe values for all PMTs

    """

    def make_quality_function(fitted_rates, rates, weights, combs):
        def quality_function(qes):
            sq_sum = 0
            for fitted_rate, comb, rate, weight in zip(
                fitted_rates, combs, rates, weights
            ):
                sq_sum += (
                    (rate / qes[comb[0]] / qes[comb[1]] - fitted_rate) * weight
                ) ** 2
            return sq_sum

        return quality_function

    qfunc = make_quality_function(fitted_rates, rates, weights, combs)
    qes = np.ones(31)
    bounds = [(0.1, 2.0)] * 31
    opt_qes = optimize.minimize(qfunc, qes, bounds=bounds)
    return opt_qes


def correct_means(means, opt_t0s, combs):
    """Applies optimal t0s to gaussians means.

    Should be around zero afterwards.

    Parameters
    ----------
    means: numpy array of means of gaussians of all PMT combinations
    opt_t0s: numpy array of optimal t0 values for all PMTs
    combs: pmt combinations used to correct

    Returns
    -------
    corrected_means: numpy array of corrected gaussian means for all PMT combs

    """
    corrected_means = np.array(
        [
            (opt_t0s[comb[1]] - opt_t0s[comb[0]]) - mean
            for mean, comb in zip(means, combs)
        ]
    )
    return corrected_means


def correct_rates(rates, opt_qes, combs):
    """Applies optimal qes to rates.

    Should be closer to fitted_rates afterwards.

    Parameters
    ----------
    rates: numpy array of rates of all PMT combinations
    opt_qes: numpy array of optimal qe values for all PMTs
    combs: pmt combinations used to correct

    Returns
    -------
    corrected_rates: numpy array of corrected rates for all PMT combinations
    """

    corrected_rates = np.array(
        [rate / opt_qes[comb[0]] / opt_qes[comb[1]] for rate, comb in zip(rates, combs)]
    )
    return corrected_rates


def calculate_rms_means(means, corrected_means):
    """Calculates RMS of means from zero before and after correction

    Parameters
    ----------
    means: numpy array of means of gaussians of all PMT combinations
    corrected_means: numpy array of corrected gaussian means for all PMT combs

    Returns
    -------
    rms_means: RMS of means from zero
    rms_corrected_means: RMS of corrected_means from zero
    """
    rms_means = np.sqrt(np.mean((means - 0) ** 2))
    rms_corrected_means = np.sqrt(np.mean((corrected_means - 0) ** 2))
    return rms_means, rms_corrected_means


def calculate_rms_rates(rates, fitted_rates, corrected_rates):
    """Calculates RMS of rates from fitted_rates before and after correction

    Parameters
    ----------
    rates: numpy array of rates of all PMT combinations
    corrected_rates: numpy array of corrected rates for all PMT combinations

    Returns
    -------
    rms_rates: RMS of rates from fitted_rates
    rms_corrected_rates: RMS of corrected_ratesrates from fitted_rates
    """
    rms_rates = np.sqrt(np.mean((rates - fitted_rates) ** 2))
    rms_corrected_rates = np.sqrt(np.mean((corrected_rates - fitted_rates) ** 2))
    return rms_rates, rms_corrected_rates


@jit
def get_comb_index(i, j):
    """Return the index of PMT pair combinations"""
    return i * 30 - i * (i + 1) // 2 + j - 1


@jit
def add_to_twofold_matrix(times, tdcs, mat, tmax=10):
    """Add counts to twofold coincidences for a given `tmax`.

    Parameters
    ----------
    times: np.ndarray of hit times (int32)
    tdcs: np.ndarray of channel_ids (uint8)
    mat: ref to a np.array((465, tmax * 2 + 1))
    tmax: int (time window)

    Returns
    -------
    mat: coincidence matrix (np.array((465, tmax * 2 + 1)))

    """
    h_idx = 0  # index of initial hit
    c_idx = 0  # index of coincident candidate hit
    n_hits = len(times)
    multiplicity = 0
    while h_idx <= n_hits:
        c_idx = h_idx + 1
        if (c_idx < n_hits) and (times[c_idx] - times[h_idx] <= tmax):
            multiplicity = 2
            c_idx += 1
            while (c_idx < n_hits) and (times[c_idx] - times[h_idx] <= tmax):
                c_idx += 1
                multiplicity += 1
            if multiplicity != 2:
                h_idx = c_idx
                continue
            c_idx -= 1
            h_tdc = tdcs[h_idx]
            c_tdc = tdcs[c_idx]
            h_time = times[h_idx]
            c_time = times[c_idx]
            if h_tdc != c_tdc:
                dt = int(c_time - h_time)
                if h_tdc > c_tdc:
                    mat[get_comb_index(c_tdc, h_tdc), -dt + tmax] += 1
                else:
                    mat[get_comb_index(h_tdc, c_tdc), dt + tmax] += 1
        h_idx = c_idx


# jmonitork40_comb_indices =  \
#     np.array((254, 423, 424, 391, 392, 255, 204, 205, 126, 120, 121, 0,
#     22, 12, 80, 81, 23, 48, 49, 148, 150, 96, 296, 221, 190, 191, 297, 312,
#     313, 386, 355, 132, 110, 431, 42, 433, 113, 256, 134, 358, 192, 74,
#     176, 36, 402, 301, 270, 69, 384, 2, 156, 38, 178, 70, 273, 404, 302,
#     77, 202, 351, 246, 440, 133, 262, 103, 118, 44, 141, 34, 4, 64, 30,
#     196, 91, 172, 61, 292, 84, 157, 198, 276, 182, 281, 410, 381, 289,
#     405, 439, 247, 356, 102, 263, 119, 140, 45, 35, 88, 65, 194, 31,
#     7, 60, 173, 82, 294, 158, 409, 277, 280, 183, 200, 288, 382, 406,
#     212, 432, 128, 388, 206, 264, 105, 72, 144, 52, 283, 6, 19, 14,
#     169, 24, 310, 97, 379, 186, 218, 59, 93, 152, 317, 304, 111, 387,
#     129, 207, 104, 265, 73, 18, 53, 5, 284, 146, 168, 15, 308, 26,
#     98, 92, 187, 58, 219, 380, 316, 154, 305, 112, 434, 257, 357, 135,
#     193, 300, 177, 401, 37, 75, 68, 271, 1, 385, 159, 403, 179, 272,
#     71, 39, 76, 303, 203, 213, 393, 248, 442, 298, 145, 184, 89, 377,
#     315, 216, 57, 309, 27, 99, 8, 54, 16, 171, 287, 153, 21, 78,
#     394, 441, 249, 299, 314, 185, 376, 90, 147, 56, 217, 25, 311, 100,
#     286, 55, 170, 17, 9, 20, 155, 79, 425, 426, 383, 306, 220, 290,
#     291, 307, 188, 189, 149, 151, 101, 86, 13, 50, 51, 87, 28, 29,
#     3, 352, 399, 375, 274, 407, 197, 285, 180, 279, 83, 295, 160, 199,
#     66, 174, 63, 33, 10, 95, 40, 400, 282, 275, 195, 408, 378, 278,
#     181, 293, 85, 161, 32, 67, 62, 175, 201, 94, 11, 41, 435, 415,
#     359, 360, 436, 347, 348, 258, 259, 318, 136, 162, 222, 223, 137, 114,
#     115, 43, 451, 443, 266, 389, 335, 456, 208, 396, 363, 250, 238, 327,
#     235, 107, 130, 215, 116, 343, 344, 452, 461, 462, 331, 332, 417, 226,
#     324, 371, 372, 229, 240, 241, 163, 142, 267, 230, 412, 122, 428, 319,
#     353, 227, 340, 166, 47, 108, 253, 138, 444, 411, 231, 427, 123, 320,
#     46, 228, 165, 341, 354, 252, 109, 139, 455, 336, 395, 209, 364, 106,
#     239, 234, 328, 251, 214, 131, 117, 373, 447, 243, 418, 164, 369, 325,
#     460, 342, 329, 237, 224, 242, 448, 419, 339, 370, 459, 326, 167, 236,
#     330, 225, 127, 365, 124, 333, 244, 450, 430, 397, 211, 260, 366, 429,
#     334, 449, 245, 125, 210, 398, 261, 321, 420, 421, 422, 322, 367, 368,
#     323, 345, 413, 232, 143, 268, 446, 361, 463, 464, 346, 453, 454, 416,
#     374, 233, 337, 458, 349, 414, 457, 338, 350, 445, 269, 362, 390, 437,
#     438))
#
"""
jmonitork40_comb_indices =  \
    np.array((417, 418, 419, 420, 421, 422, 363, 364, 365, 366, 367, 368,
    318, 319, 320, 321, 322, 323, 156, 157, 158, 159, 160, 161, 96, 97, 98,
    99, 100, 101, 461, 369, 324, 371, 464, 427, 331, 237, 238, 333, 434, 415,
    339, 231, 175, 232, 342, 278, 184, 61, 62, 186, 281, 220, 162, 54, 13,
    56, 168, 459, 370, 325, 374, 423, 428, 328, 244, 239, 338, 343, 411, 346,
    226, 178, 229, 270, 271, 181, 68, 69, 191, 170, 216, 164, 50, 16, 58,
    462, 373, 326, 327, 429, 424, 337, 240, 245, 222, 345, 412, 347, 228, 179,
    180, 272, 273, 190, 70, 71, 48, 163, 217, 172, 57, 17, 463, 372, 234,
    332, 430, 431, 334, 241, 174, 230, 340, 416, 341, 233, 60, 185, 279, 280,
    187, 63, 12, 52, 165, 221, 166, 59, 460, 242, 235, 336, 425, 432, 330,
    223, 176, 225, 348, 413, 350, 64, 65, 189, 274, 275, 183, 49, 14, 55,
    173, 218, 169, 335, 236, 243, 329, 433, 426, 344, 224, 177, 227, 349, 414,
    188, 66, 67, 182, 276, 277, 171, 53, 15, 51, 167, 219, 387, 204, 128,
    209, 396, 443, 435, 263, 112, 120, 249, 375, 283, 73, 6, 85, 301, 310,
    306, 148, 22, 28, 154, 388, 208, 126, 211, 254, 451, 452, 256, 104, 105,
    282, 383, 285, 84, 1, 87, 144, 312, 313, 150, 24, 25, 395, 210, 129,
    110, 262, 436, 445, 248, 121, 72, 284, 376, 300, 86, 7, 18, 146, 307,
    314, 152, 29, 389, 205, 111, 118, 247, 446, 437, 265, 4, 81, 299, 377,
    287, 75, 19, 26, 149, 315, 308, 155, 390, 255, 102, 103, 257, 453, 454,
    80, 0, 83, 286, 384, 289, 145, 20, 21, 151, 316, 317, 444, 246, 119,
    113, 264, 438, 298, 82, 5, 74, 288, 378, 311, 147, 27, 23, 153, 309,
    351, 136, 42, 138, 354, 403, 194, 33, 34, 200, 410, 385, 292, 91, 3,
    94, 297, 359, 137, 44, 133, 399, 404, 196, 40, 35, 202, 290, 379, 303,
    92, 10, 79, 352, 132, 45, 192, 405, 400, 198, 36, 41, 88, 302, 380,
    294, 78, 11, 353, 139, 30, 195, 406, 407, 201, 37, 2, 90, 293, 386,
    296, 95, 360, 38, 31, 197, 401, 408, 203, 89, 8, 77, 295, 381, 305,
    193, 32, 39, 199, 409, 402, 291, 76, 9, 93, 304, 382, 355, 134, 46,
    141, 362, 455, 439, 251, 108, 124, 269, 356, 140, 43, 143, 258, 447, 448,
    260, 116, 117, 361, 142, 47, 106, 250, 440, 457, 268, 125, 357, 135, 107,
    122, 267, 458, 441, 253, 358, 259, 114, 115, 261, 449, 450, 456, 266, 123,
    109, 252, 442, 391, 212, 127, 214, 394, 397, 213, 130, 207, 392, 206, 131,
    393, 215, 398))
"""
