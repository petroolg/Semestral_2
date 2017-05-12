"""
Functions for inference in HMMs

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from collections import Counter
from utils import normalized
import numpy as np

def update_belief_by_time_step(prev_B, hmm):
    """Update the distribution over states by 1 time step.

    :param prev_B: Counter, previous belief distribution over states
    :param hmm: contains the transition model hmm.pt(from,to)
    :return: Counter, current (updated) belief distribution over states
    """
    cur_B = Counter()
    for x in prev_B:
        for prev_x in prev_B:
            cur_B[x] = hmm.pt(prev_x, x) * prev_B [prev_x] + cur_B[x]
        
            
    alpha = 0
    for x in cur_B:
        alpha = alpha + cur_B[x]
    for x in cur_B:
        cur_B[x] = cur_B[x]/alpha
    return cur_B


def predict(n_steps, prior, hmm):
    """Predict belief state n_steps to the future

    :param n_steps: number of time-step updates we shall execute
    :param prior: Counter, initial distribution over the states
    :param hmm: contains the transition model hmm.pt(from, to)
    :return: sequence of belief distributions (list of Counters),
             for each time slice one belief distribution;
             prior distribution shall not be included
    """
    B = prior  # This shall be iteratively updated
    Bs = [B]    # This shall be a collection of Bs over time steps

    for i in range(0, n_steps):
        Bs.append(update_belief_by_time_step(Bs[i], hmm))    
    return Bs


def update_belief_by_evidence(prev_B, e, hmm, normalize=False):
    """Update the belief distribution over states by observation

    :param prev_B: Counter, previous belief distribution over states
    :param e: a single evidence/observation used for update
    :param hmm: HMM for which we compute the update
    :param normalize: bool, whether the result shall be normalized
    :return: Counter, current (updated) belief distribution over states
    """
    # Create a new copy of the current belief state
    cur_B = Counter(prev_B)
    # Your code here
    for x in hmm.cur_B:
        cur_B[x] = prev_B[x] * hmm.pe(x, e)
    if(normalize == True): 
        total = sum(cur_B.values(), 0.0)
        for key in cur_B:
            cur_B[key] /= total
    return cur_B


def forward1(prev_f, cur_e, hmm, normalize=False):
    """Perform a single update of the forward message

    :param prev_f: Counter, previous belief distribution over states
    :param cur_e: a single current observation
    :param hmm: HMM, contains the transition and emission models
    :param normalize: bool, should we normalize on the fly?
    :return: Counter, current belief distribution over states
    """
    # Your code here
    cur_f = Counter()
    for xt in prev_f:
        for xtp in prev_f:
            cur_f[xt] = cur_f[xt] + hmm.pt(xtp,xt)*prev_f[xtp]
        cur_f[xt] = cur_f[xt] * hmm.pe(xt, cur_e)
        
    if(normalize == True): 
        total = sum(cur_f.values(), 0.0)
        for key in cur_f:
            cur_f[key] /= total
    return cur_f


def forward(init_f, e_seq, hmm):
    """Compute the filtered belief states given the observation sequence

    :param init_f: Counter, initial belief distribution over the states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: sequence of Counters, estimates of belief states for all time slices
    """
    f = init_f    # Forward message, updated each iteration
    fs = []       # Sequence of forward messages, one for each time slice
    # Your code here
    fs.append(f)
    for e in e_seq:
        fs.append(forward1(fs[-1], e, hmm, normalize = True))
    return fs[1:]


def likelihood(prior, e_seq, hmm):
    """Compute the likelihood of the model wrt the evidence sequence

    In other words, compute the marginal probability of the evidence sequence.
    :param prior: Counter, initial belief distribution over states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: number, likelihood
    """
    l_t = Counter(Counter())
    l_t = prior
            
    for obs in e_seq:
        l_t = forward1(l_t, obs, hmm)        
        
    L_t = 0
    for state in hmm.X_domain:
        L_t = L_t + l_t[state]
    lhood = L_t
    return lhood

def backward1(next_b, next_e, hmm):
    """Propagate the backward message

    :param next_b: Counter, the backward message from the next time slice
    :param next_e: a single evidence for the next time slice
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current backward message
    """
    cur_b = Counter()
    # Your coude here
    for state_curr in next_b:
        for state_next in next_b:
            cur_b[state_curr] = cur_b[state_curr] + hmm.pe(state_next, next_e) * hmm.pt(state_curr, state_next) * next_b[state_next]
    return cur_b

def forwardbackward(priors, e_seq, hmm):
    """Compute the smoothed belief states given the observation sequence

    :param priors: Counter, initial belief distribution over rge states
    :param e_seq: sequence of observations
    :param hmm: HMM, contians the transition and emission models
    :return: sequence of Counters, estimates of belief states for all time slices
    """
    se = list()   # Smoothed belief distributions
    f = list()
    f.append(priors)
    f.extend(forward(priors, e_seq, hmm))
    b = Counter()
    prod = Counter()
    for state in priors:
        b[state] = 1
    e_seq.reverse()
    i = len(e_seq)
    for obs in e_seq:
        #vypocet f_(i-1) x b
        for x in priors:
            prod[x] = f[i][x] * b[x]
        #vypocet s_i
        se.append(normalized(prod))
        #vypocet b
        b = backward1(b, obs, hmm)   
        i = i - 1
    se.reverse()
    return se

def viterbi1(prev_m, cur_e, hmm):
    """Perform a single update of the max message for Viterbi algorithm

    :param prev_m: Counter, max message from the previous time slice
    :param cur_e: current observation used for update
    :param hmm: HMM, contains transition and emission models
    :return: (cur_m, predecessors), i.e.
             Counter, an updated max message, and
             dict with the best predecessor of each state
    """
    cur_m = Counter()   # Current (updated) max message
    predecessors = {}   # The best of previous states for each current state
    # Your code here
    
    
    # Find max P(X_t | x_t-1)
    for xt in prev_m:
        m = -1
        best_xtp = None
        for xtp in prev_m:
            if hmm.pt(xtp, xt) * prev_m[xtp] > m:
                m = hmm.pt(xtp, xt) * prev_m[xtp]
                best_xtp = xtp
        cur_m[xt] = hmm.pe(xt, cur_e) * m
        predecessors[xt] = best_xtp
  
    #total = sum(cur_m.values(), 0.0)
    #for key in cur_m:
    #    cur_m[key] /= total
    
    return cur_m, predecessors


def viterbi(priors, e_seq, hmm):
    """Find the most likely sequence of states using Viterbi algorithm

    :param priors: Counter, prior belief distribution
    :param e_seq: sequence of observations
    :param hmm: HMM, contains the transition and emission models
    :return: (sequence of states, sequence of max messages)
    """
    ml_seq = []  # Most likely sequence of states
    ms = []      # Sequence of max messages
    bestPredecessors = []
    # Your code here
    ms.append(forward1(priors, e_seq[0], hmm))

    for obs in e_seq[1:]:
        cur_m, predecessors = viterbi1(ms[-1], obs, hmm)
        ms.append(cur_m)
        bestPredecessors.append(predecessors)

    m = -1
    finalState = None
    for key in ms[-1]:
        if(ms[-1][key] > m):
            finalState = key
            m = ms[-1][key]

    ml_seq = viterbi_ML(finalState, bestPredecessors)

    return ml_seq, ms

def viterbi_bestPrevState(currState, bestPredecessors):
    "Find most likely predecessors from current state with dict of best prev. stats"
    return bestPredecessors[currState]

def viterbi_ML(finalState, bestPredecessors):
    "Find most likely sequence which ends in finalState"
    bp = bestPredecessors.copy()
    bp.reverse()
    mlseq = []
    mlseq.append(finalState)
    for bestPred in bp:
        mlseq.append(viterbi_bestPrevState(mlseq[-1], bestPred))
    # mlseq.reverse()
    return mlseq