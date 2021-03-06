#!/bin/env python
'''
File with adapter for inference with external library yahmm. Constructor of class Adapter
converts Robot chain into graph, which is than used for inference.
'''
import numpy as np
from collections import Counter
from itertools import combinations_with_replacement, permutations, product
from yahmm import *

obsrv_prod = list(product(['n', 'f'], ['n', 'f'], ['n', 'f'], ['n', 'f']))

def mergeChars(str):
    merged = list()
    for obs in str:
        merged.append(''.join(obs))
    return merged

obsrv = mergeChars(obsrv_prod)

class Adapter():
    '''
    Usage: contructor converts all parameters from robot.py into yahmm form. Then you can use provided
    functions for inference.
    '''
    def __init__(self, robot, initBelief_counter):
        '''
        :param robot: instance of Robot as defined in  robot.py
        :param initBelief: Counter, initial belief over states (but it is not actually used here, we
        assume that we know nothing about robot position and use uniform distribution over all states)
        '''
        freePos = robot.maze.get_free_positions(search = True)
        pt = {}
        pe = {}
        dists = list()
        initP = 1/len(freePos)

        model = Model("Robot")
        #Create dicts
        for state in freePos:
            pe[state] = {}
            pt[state] = {}
            for nextState in freePos:
                pt[state][nextState] = robot.pt(state, nextState)
            for obs in obsrv:
                pe[state][obs] = robot.pe(state, obs)

        # Create states
        states = list()
        for state in freePos:
            states.append(State(DiscreteDistribution(pe[state])))

        # Add them to model
        for state in states:
            model.add_state(state)
        # Initial states
        for state in states:
            model.add_transition(model.start, state, initP)
        # Connections in model
        for state1 in states:
            for state2 in states:
                model.add_transition(state1,
                                     state2,
                                     float(pt[freePos[states.index(state1)]][freePos[states.index(state2)]]))

        model.bake(verbose=True)
        self.model = model
        self.freePos = freePos
        self.states = states


    def forward(self, obsSeq):
        '''
        :param obsSeq: sequence of observations
        :return: list of Counters (beliefs) over states for each observation
        '''
        obsSeq = mergeChars(obsSeq)
        mat = self.model.forward(obsSeq)
        mat = np.exp(mat)

        for row in range(0,mat.shape[0]):
            mat[row,:] = mat[row,:]/np.sum(mat[row,:])

        mat = mat[1:,0:-2]
        beliefs = list()
        for i, obs in enumerate(obsSeq):
            beliefs.append(Counter(self.freePos))
            for j,state in enumerate(beliefs[i]):
                beliefs[i][state] = mat[i,j]
        return beliefs

    def forwardbackward(self, obsSeq):
        '''
        :param obsSeq: sequence of observations
        :return: list of Counters, one for each observation
        '''
        obsSeq = mergeChars(obsSeq)
        mat = self.model.forward_backward(obsSeq)
        mat = np.exp(np.array(mat[1]))
        for row in range(0,mat.shape[0]):
            mat[row,:] = mat[row,:]/np.sum(mat[row,:])

        beliefs2 = list()
        for i,obs in enumerate(obsSeq):
            beliefs2.append(Counter())
            for j,state in enumerate(self.freePos):
                beliefs2[i][state] = mat[i,j]
        return beliefs2

    def viterbi(self, obsSeq):
        '''
        :param obsSeq: sequence of observations
        :return: most likely sequence, sequence of most likely states is returned as None,
        just to be compatible with hmm_inference... we dont need it for our inference here...
        '''
        obsSeq = mergeChars(obsSeq)
        mls = list()
        for i in range(1,len(obsSeq)+1):
            ml_state = self.model.viterbi(obsSeq[0:i])
            ml_state = ml_state[1][-1]
            mls.append(self.freePos[self.states.index(ml_state[1])])
        ms = None
        return mls, ms
