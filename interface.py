#!/bin/env python
from pomegranate import HiddenMarkovModel, DiscreteDistribution, Distribution
import numpy as np
from collections import Counter
from itertools import combinations_with_replacement, permutations, product

obsrv = list(product(['n', 'f'], ['n', 'f'], ['n', 'f'], ['n', 'f']))

class Adapter():
    '''
    Usage: contructor converts all parameters from robot.py into pomegranate form. Then you can use provided
    functions for inference...
    '''
    def __init__(self, robot, initBelief_counter, obs_seq):
        '''
        :param robot: instance of Robot as defined in  robot.py
        :param initBelief: Counter, initial belief over states
        '''
        freePos = robot.maze.get_free_positions(search = True)
        pt = Counter()
        pe = Counter()
        dists = list()
        trans_mat = np.zeros((len(freePos)**1, len(freePos)**1))
        initBelief = np.array(np.ones(len(freePos))/len(freePos))

        # translate obs_seq into number form
        obs_seq_translated = list()
        for i in range(0, len(obs_seq)):
            for j, obs in enumerate(obsrv):
                if obs == obs_seq[i]:
                    obs_seq_translated.append(j)

        i = 0;j = 0;
        for curPos in freePos:
            pt[curPos] = {}
            pe[curPos] = {}
            initBelief[i] = initBelief_counter[curPos]
            j = 0
            for nextPos in freePos:
                pt[curPos][nextPos] = robot.pt(curPos, nextPos)
                trans_mat[i,j] = robot.pt(curPos, nextPos)
                j = j + 1
            for k,obs in enumerate(obsrv):
                pe[curPos][k] = robot.pe(curPos, obs)
                #print(robot.pe(curPos, obs))
            dists.append(DiscreteDistribution(pe[curPos]))
            i = i + 1

        self.trans_mat = trans_mat
        self.dists = dists
        self.initBelief = initBelief
        self.obs_seq = obs_seq_translated
        self.freePos = freePos

    def forward(self):
        '''
        :return: list of Counters, one for each observation,
        each of them contains probabilities of being on every position in maze
        But something is probably broken... Maybe mysterious matrix returned from hmm.forward
        (https://pomegranate.readthedocs.io/en/latest/HiddenMarkovModel.html)... - It seems to be
        larger then it sould be according to dcs!!!! And first line is always full of -Inf and one zero...
        '''
        hmm = HiddenMarkovModel.from_matrix(self.trans_mat, self.dists, starts= self.initBelief)
        mat = hmm.forward(self.obs_seq)
        #mat = mat[1:,0:-2]
        #print(mat)
        beliefs = list()
        #mat[np.isinf(-mat)] = nan
        for i, obs in enumerate(self.obs_seq):
            #print(i)
            beliefs.append(Counter(self.freePos))
            for j,state in enumerate(beliefs[i]):
                #print(j)
                beliefs[i][state] = 10**(mat[i,j])
        return beliefs

    def forwardbackward(self):
        '''
        :return: list of Counters, one for each observation,
        each of them contains probabilities of being on every position in maze
        It seems it works... +-... It is a bit different than our implementation, but not that much as forward
        algorithm... Maybe after row denormalization and removal of logs it will be ok?
        '''
        hmm = HiddenMarkovModel.from_matrix(self.trans_mat, self.dists, starts= self.initBelief)
        matT = hmm.forward_backward(self.obs_seq)
        mat2 = matT[0]
        mat = np.array(matT[1])

        beliefs = list()
        for i, obs in enumerate(self.obs_seq):
            beliefs.append(Counter(self.freePos))
            for j, state in enumerate(beliefs[i]):
                beliefs[i][state] = (mat[i][j])
        return beliefs