from scipy.spatial import distance

from robot import *
from hmm_inference import *
import numpy as np
import scipy
from scipy import spatial as sp
import matplotlib.pyplot as plt
import numpy as np
import os

N_STEPS = 20
SCALE =20
MAX_DIST = 7
N_DIST = MAX_DIST*SCALE
#weighted Manhattan distances from real positions
dev = [[0]*N_DIST,[0]*N_DIST,[0]*N_DIST]

for maze in os.listdir('mazes/'):

    m = Maze('mazes/'+maze)
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
        for p in f.most_common():
            if p[1] == 0:
                break
            dist += sp.distance.cityblock(p[0], s)*p[1]
        dev[0][int(dist*SCALE)] += 1

        dist = 0
        for p in fb.most_common():
            if p[1] == 0:
                break
            dist += sp.distance.cityblock(p[0], s) * p[1]
        dev[1][int(dist*SCALE)] += 1

        dist = sp.distance.cityblock(v, s)
        dev[2][int(dist*SCALE)] += 1

x = list(np.linspace(0,MAX_DIST,num = N_DIST))
width =1/1.5
fig, ax = plt.subplots(nrows=1,ncols=3, sharey=True, figsize=(9,2))
plt.tight_layout()
titles = ['Forward algorithm', 'Forward-backward algorithm','Viterbi algorithm']

for i in range(0,3):
    plt.subplot(1, 3, (i+1))
    plt.bar(x, dev[i], width)
    plt.title(titles[i])
    plt.xlabel('Weighted Manhattan distance from real position')
    plt.ylabel('Frequency')

plt.show()