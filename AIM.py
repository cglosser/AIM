import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from collections import namedtuple

BasisFunction = namedtuple("BasisFunction", ["begin", "end", "mid"])

class GridCoordinate(object):
    def __init__(self, numCols, pt):
        self.numCols = numCols
        self.pt      = np.array(pt).astype(int)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.numCols != other.numCols:
                raise Exception("Column mismatch in GridCoordinate addition")
            return GridCoordinate(self.numCols, self.pt + other.pt)
        else:
            npOther = np.array(other).astype(int)
            return GridCoordinate(self.numCols, self.pt + npOther)

    def __str__(self):
        return str(self.pt)

    def toIndex(self):
        return self.pt[1]*self.numCols + self.pt[0]

class Grid(object):
    def __init__(self, gridLines, size = 1):
        self.size = size
        self.gridDim = gridLines
        self.numPts  = gridLines**2
        self.gridSpacing = size/float(gridLines - 1) #step size
        self.boxes = None
        self.gridPts = np.array([(x,y) for y in range(self.gridDim)
            for x in range(self.gridDim)])/float(self.gridDim - 1)
        
    def __boundsCheck(self, pt):
        """
        Check that all x,y pairs lie within [0, 1]. Should make this an exception
        """
        assert np.all(pt <= self.size) and np.all(pt >= 0.0), "point lies outside the grid"

    def anchorGridCoord(self, pt):
        """
        Returns the GridCoordinate of the grid point to the south west of pt.
        """
        self.__boundsCheck(pt)
        anchor = np.floor(pt/self.gridSpacing).astype(int)
        return GridCoordinate(self.gridDim, anchor)
    
    def boxGridCoords(self, pt):
        """
        Return the four GridCoordinates of the box enclosing pt
        """
        anchor = self.anchorGridCoord(pt)
        return [anchor + [dx, dy] for dy in range(0, 2) for dx in range(0, 2)]

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

def sample_to_basis(pts):
    """Build a collection of rect (pulse) basis functions.
    
    Given a closed set of input points, convert each adjacent pair to a rect 
    basis function defined by begining (given), end (given), and mid (average 
    of begining and end) points.
    """
    return [BasisFunction(begin=x1, end=x2, mid=(x1 + x2)/2.0) 
        for x1, x2 in zip(pts, np.roll(pts, -1, axis = 0))]

def q_matrix_element(m_vec, basis_func):
    if np.all(m_vec == 0):
        return 1 #integrating 1 from t = 0 to t = 1
    func = lambda t: np.prod(np.power((1-t)*basis_func.begin + 
        t*basis_func.end-basis_func.mid, m_vec))
    return quad(func, 0, 1)

def w_matrix_element(m_vec, basis_func, u_vec):
    return np.prod(np.power(u_vec - basis_func.mid, m_vec))

class Linktable(object):
    def __init__(self, grid, pts):
        self.grid = grid; self.pts = pts
        self.gridLookup  = [grid.nearestGridIdx(p) for p in pts]
        self.pointLookup = [[] for _ in range(grid.numPts)]
        for idx, p in enumerate(self.gridLookup):
            self.pointLookup[p].append(idx)
