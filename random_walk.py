import numpy as np
import itertools
import collections
import matplotlib.pyplot as plt
import random

'''
Reflection upon solution:
    a) Den første opgave var rimelig nem at løse
    b) Det svære i opgaven var at regne ud hvordan at matricen skulle se ud, samt hvordan 
    at funktionen imshow virkede. Herudover skulle vi reverse vores range iterator, idet 
    vi ellers ikke vil få det rigtige ud.
'''

#### a)
def random_walk():
    cur = (0, 0)
    yield cur
    while True:
        x, y = 0, 0
        while x == 0 or y == 0:
            # continue generating random values for x and y until one of them is non-zero
            x = random.choice([-1, 0, 1, 0])
            y = random.choice([-1, 0, 1, 0])
        cur = cur[0] + x, cur[1] + y
        yield cur

#### b)
# Simulate random walk
L = list(itertools.islice(random_walk(), 0, 100000))
counter = collections.Counter(L) # Count occurences of points

# Find bounds
minx = min(L)[0]
maxx = max(L)[0]
miny = min(L, key=lambda coord: coord[1])[1]
maxy = max(L, key=lambda coord: coord[1])[1]

#go over 1 rows, then move over columns in that row. Reverse range, to go in the "right" direction, else we move down the y-axis.
matrix = np.array([[counter[i,j] if (i,j) in counter else 0 for i in range(minx, maxx + 1)] for j in range(maxy, miny - 1, -1)])

# Plotting
plt.imshow(matrix, cmap = "jet", extent=(minx, maxx, miny, maxy))
plt.colorbar()
plt.show()
