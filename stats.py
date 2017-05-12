from scipy.spatial import distance

from robot import *
from hmm_inference_yahmm import *
import numpy as np
import scipy
from scipy import spatial as sp
import matplotlib.pyplot as plt
import numpy as np
import os

N_STEPS = 100
N_MAZES =2
#weighted Manhattan distances from real positions
dev = [[],[],[]]
acc = [0, 0, 0]
path = 'mazes/big_obst/'
titles = ['Forward', 'Forward-backward','Viterbi']

for maze in os.listdir(path):

    m = Maze(path+maze)
    print(maze)
    open_pos = []
    for row in range(0,m.height):
        for cell in range(0,m.width):
            if m.map[row][cell] == m.FREE_CHAR:
                open_pos.append((row,cell))

    init_f = Counter(open_pos)

    for elem in init_f.elements():
        init_f[elem] = 1/len(init_f.items())

    robot = Robot()
    robot.maze = m
    robot.position = (1, 1)
    states, observations = robot.simulate(n_steps=N_STEPS)

    forw = forward(init_f, observations, robot)
    forw_backw = forwardbackward(init_f, observations, robot)
    vtrb = viterbi(init_f, observations, robot)

    print("predictions")
    for i, (f, fb, v, s, o) in enumerate(zip(forw, forw_backw, vtrb[0], states, observations)):
    # for i, (f, s, o) in enumerate(zip(forw, states, observations)):
        # print('Step:', i + 1, '| State:', s, '| Observation:', o)
        # print("Forward algorithm")
        # for p in f.most_common()[:4]:
        #     print(p)
        # print("Forward-backward algorithm")
        # for p in fb.most_common()[:4]:
        #     print(p)
        # print("Viterbi algorithm")
        # print(v)
        # print("\n")

        dist = 0
        mc = f.most_common()
        for p in f.most_common():
            if p[1] == 0:
                break
            dist += sp.distance.cityblock(p[0], s)*p[1]
            if dist == 0 and p[1] == (f.most_common())[0][1]:
                acc[0] += 1
        dev[0].append(dist)

        dist = 0
        for p in fb.most_common():
            if p[1] == 0:
                break
            dist += sp.distance.cityblock(p[0], s) * p[1]
            if dist == 0 and p[1] == (fb.most_common())[0][1]:
                acc[1] += 1
        dev[1].append(dist)

        dist = sp.distance.cityblock(v, s)
        dev[2].append(dist)
        if dist ==0:
            acc[2] += 1

#accuracy
acc = np.array(acc) / (N_MAZES *N_STEPS)

#distributions & percentile
fig, ax = plt.subplots(nrows=1,ncols=3, sharex=True, figsize=(8,1))
plt.tight_layout()

for i in range(0,3):
    print('Distribution of', titles[i], 'algorithm: mean =', "%.2f" % np.mean(dev[i]), ', std =', "%.2f" % np.std(dev[i]))
    print('Accuracy of', titles[i], 'algorithm is', "%.2f" % acc[i])
    print('Radius of reliability area for', titles[i], 'algorithm is',"%.2f" % np.percentile(dev[i],90))
    plt.subplot(1,3, (i+1))
    # plt.bar(x, dev[i], width)
    weights = np.ones_like(dev[i]) / float(len(dev[i]))
    plt.hist(dev[i],weights=weights)
    plt.title(titles[i])
    plt.xlabel('Weighted Manhattan distance from real position')
    plt.ylabel('Frequency of occurrence')
    print('\n')
plt.show()