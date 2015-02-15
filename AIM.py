import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

class Grid(object):
    def __init__(self, gridLines, size = 1):
        self.size = size
        self.gridDim = gridLines
        self.numPts  = gridLines**2
        self.gridSpacing = size/float(gridLines - 1) #step size
        self.boxes = None
        self.gridPts = np.array([(x,y) for y in range(self.gridDim)
            for x in range(self.gridDim)])/float(self.gridDim - 1)
        
    def boxIndexToCoord(self, idx):
        assert idx >= 0, "invalid box index"
        row = idx//self.gridDim
        col = idx - row*self.gridDim
        return np.array([row, col])
        
    def boxCoordToIndex(self, coord):
        row, col = coord
        return col*self.gridDim + row
    
    def nearestGridCoord(self, pt):
        """
        Get the x,y pair of the nearest gridpoint in integral grid coordinates
        """
        assert np.all(pt <= self.size) and np.all(pt >= 0.0), "point lies outside of the grid"
        return np.round(pt/self.gridSpacing).astype(int)

    def nearestGridIdx(self, pt):
        assert np.all(pt <= self.size) and np.all(pt >= 0.0), "point lies outside of the grid"
        return self.boxCoordToIndex(self.nearestGridCoord(pt))

    def drawPoints(self, ax = None):
        if ax is None: ax = plt.gca()
        x, y = zip(*self.gridPts)
        ax.scatter(x, y, s=100, alpha=0.2)
        return ax
    
    def drawLines(self, ax = None):
        if ax is None: ax = plt.gca()
        x, y = zip(*self.gridPts)
        ax.set_xticks(x); ax.set_yticks(y)
        ax.grid()
        return ax

    def annotatePoints(self, ax = None):
        if ax is None: ax = plt.gca()
        for idx, pt in enumerate(self.gridPts): ax.annotate(idx, pt)
        return ax

        
def drawObjectAndGrid(pts, grid):
    fig = plt.figure(); f1 = fig.add_subplot(111)

    x_obj,  y_obj  = zip(*pts)
    x_grid, y_grid = zip(*grid.gridPts)
    
    f1.scatter(x_obj, y_obj, s=25, c='r', alpha=0.8)
    f1.scatter(x_grid, y_grid, s=100, alpha=0.2)
    
def drawObjectToGrid(grid, pts, ax = None):
    if ax is None: ax = plt.gca()
    
    x_obj,  y_obj  = zip(*pts)
    x_grid, y_grid = zip(*grid.gridPts)

    ax.scatter(x_obj, y_obj, c='r', s=25, alpha=0.8)

    for p in pts:
        nearestGrid = grid.nearestGridCoord(p)*grid.gridSpacing
        ax.plot([p[0], nearestGrid[0]], [p[1], nearestGrid[1]], 'r')

def sampleCircle(npts):
    pts = np.array([(np.cos(t), np.sin(t)) for t in np.arange(0, 2*np.pi, 2*np.pi/npts)])

    #shift to first quadrant
    pts += np.array([1,1])
    pts /= 2.0

    return pts

def q_matrix_element(r0, ri, rf, ms):
    assert len(r0) == len(ri) == len(rf) == len(ms)
    func = lambda t: np.prod(np.power((1-t)*ri + t*rf-r0, ms))
    return quad(func, 0, 1)[0]

class Linktable(object):
    def __init__(self, grid, pts):
        self.grid = grid; self.pts = pts
        self.gridLookup  = [grid.nearestGridIdx(p) for p in pts]
        self.pointLookup = [[] for _ in range(grid.numPts)]
        for idx, p in enumerate(self.gridLookup):
            self.pointLookup[p].append(idx)
