"""
Reflection upon solution:
    1: The points can be made with a simple for loop, that outputs 2 coordinates  

    2: Plot hull takes the points and the polygon. We add the polygon[0] to make the line "come back"
    
    3: The convex hull algorithm is made with inspiration from the following wikipedia site 
    https://en.wikipedia.org/wiki/Graham_scan . The algorithm was hard to grasp at first,
    but. It was difficult understanding the logic needed to use the left turn function, and also
    that we needed to pop the second to last element. This can be viewed in the animation.
    Once the logic made sense the solution was fairly simple to write. 
"""
import random
import matplotlib.pyplot as plt


### a
def random_points(n):
    return [(round(random.uniform(0.0, 1), 5), round(random.uniform(0.0, 1), 5)) for x in range(n)]

points = random_points(25)

### b
def plot_hull(points, polygon):
    
    polygon = polygon + [polygon[0]] # To ensure final point is starting point
    plt.plot(*list(zip(*polygon)), "r-")

    plt.plot(*list(zip(*points)), "go")

    plt.plot(*list(zip(*polygon)), ".r")

    plt.show()

### c
def convex_hull(points):
    
    '''
    Return convex hull. 
    
    Examples:
    >>> convex_hull([(1,1),(2,3),(2,2),(4,2),(3,1)])
    [(1, 1), (2, 3), (4, 2), (3, 1)]

    >>> convex_hull([(4,1),(3,1),(2,2),(3,2),(3,3),(4,2)])
    [(2, 2), (3, 3), (4, 2), (4, 1), (3, 1)]
    '''

    def get_slope(p, q):
        if p == q or p[0] == q[0]:
            return float("inf")
        
        return (p[1]-q[1])/(p[0]-q[0]) 
    

    def left_turn(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]) >= 0
    
    chl = []

    chl.append((sorted_points := sorted(points, key=lambda x: (x[0], x[1])))[0])

    sorted_points = sorted(sorted_points, key = lambda p: (get_slope(p, chl[0]), p[0],p[1]))

    for point in sorted_points:
        
        chl.append(point)

        while len(chl) > 2 and left_turn(chl[-3], chl[-2], chl[-1]) == False:
            # Popper den miderste, idet at det er den der laver en bøgning til højre, som vi ikke må
            chl.pop(-2)

    # Vi skal lige remove det første element og så reverse den.
    chl = chl[1:]
    return chl[::-1]


CH = convex_hull(points)

plot_hull(points, CH)

def left_turn(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]) >= 0

### Task d
# Assertion first 
debugMode = True
if debugMode:
    # Assert Input
    for tup in points:
        assert isinstance(tup, tuple) and len(tup) == 2, "Not tuple or point"
        for num in tup:
            assert isinstance(num, (int, float)), "Not float or int"

    # Assert Output
    assert set(CH) <= set(points), "CH not a subset."

    for i in range(len(CH) - 2):
        assert not left_turn(CH[i-1], CH[i], CH[i+1]), "Not clockwise order, or not a convex point set, \
            or not smallest point set."
    assert not left_turn(CH[-2], CH[-1], CH[0]), "Not clockwise order, or a convex point set, \
            or not smallest point set."

    assert convex_hull(points)[0] == sorted(points)[0], "CH[0] is not lexicographically smallest point."

# Doctest
import doctest
doctest.testmod(verbose=True)
