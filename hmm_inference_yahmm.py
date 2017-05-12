"""
Functions for inference in HMMs
Interface for yahmm library, it has got same usage as inference functions
in hmm_inference.py
"""

from collections import Counter
from utils import normalized
import numpy as np
from hmm_interface_yahmm import Adapter


def forward(init_f, e_seq, hmm):
    """Compute the filtered belief states given the observation sequence

    :param init_f: Counter, initial belief distribution over the states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: sequence of Counters, estimates of belief states for all time slices
    """
    a = Adapter(hmm, init_f)
    fs = a.forward(e_seq)
    return fs


def forwardbackward(priors, e_seq, hmm):
    """Compute the smoothed belief states given the observation sequence

    :param priors: Counter, initial belief distribution over states
    :param e_seq: sequence of observations
    :param hmm: HMM, contians the transition and emission models
    :return: sequence of Counters, estimates of belief states for all time slices
    """
    a = Adapter(hmm, priors)
    se = a.forwardbackward(e_seq)
    return se


def viterbi(priors, e_seq, hmm):
    """Find the most likely sequence of states using Viterbi algorithm

    :param priors: Counter, prior belief distribution
    :param e_seq: sequence of observations
    :param hmm: HMM, contains the transition and emission models
    :return: (sequence of states, sequence of max messages)
    """
    a = Adapter(hmm, priors)
    ml_seq, ms = a.viterbi(e_seq)
    return ml_seq, ms