import numpy as np
import matplotlib.pyplot as plt

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
        """
        Convert a box index to integral grid coordinates
        """
        assert idx >= 0, "invalid box index"
        row = idx//self.gridDim
        col = idx - row*self.gridDim
        return np.array([row, col])
        
    def boxCoordToIndex(self, coord):
        """
        Convert two grid coordinates to a box index
        """
        row, col = coord
        return col*self.gridDim + row
    
    def nearestGridCoord(self, pt):
        """
        Return the coordinates of the nearest gridpoint as a numpy array of "grid units" (# of boxes)
        """
        assert np.all(pt <= self.size) and np.all(pt >= 0.0), "point lies outside of the grid"
        return np.round(pt/self.gridSpacing).astype(int)

    def nearestGridIdx(self, pt):
        """
        Return the index of the nearest gridpoint
        """
        assert np.all(pt <= self.size) and np.all(pt >= 0.0), "point lies outside of the grid"
        return self.boxCoordToIndex(self.nearestGridCoord(pt))
    
    def drawGrid(self, showGrid = False):
        gridFig = plt.figure()
        grid = gridFig.add_subplot(111)
        
        pts = np.array([(x,y) for y in range(self.gridDim) for x in range(self.gridDim)])
        gridPoints = pts/float(self.gridDim - 1)
        x, y = zip(*gridPoints)
        
        grid.scatter(x, y, s=100, alpha=0.2)
        if showGrid:
            grid.set_xticks(x); grid.set_yticks(y)
            grid.grid() #could have named that better...
        
        if self.gridDim < 10:
            for pt in gridPoints:
                boxCoord = self.nearestGridCoord(pt)
                grid.annotate(self.boxCoordToIndex(boxCoord), pt, xycoords="data")
        
        gridFig.show()
        
def drawObjectAndGrid(pts, grid):
    fig = plt.figure(); f1 = fig.add_subplot(111)

    x_obj,  y_obj  = zip(*pts)
    x_grid, y_grid = zip(*grid.gridPts)
    
    f1.scatter(x_obj, y_obj, s=25, c='r', alpha=0.8)
    f1.scatter(x_grid, y_grid, s=100, alpha=0.2)
    
def drawObjectToGrid(grid, pts, gridLabels=False):
    fig = plt.figure(); f1 = fig.add_subplot(111)
    
    x_obj,  y_obj  = zip(*pts)
    x_grid, y_grid = zip(*grid.gridPts)

    f1.scatter(x_obj, y_obj, s=25, c='r', alpha=0.8)
    f1.scatter(x_grid, y_grid, s=100, alpha=0.2)
    
    for p in pts:
        nearestGrid = grid.nearestGridCoord(p)*grid.gridSpacing
        f1.plot([p[0], nearestGrid[0]], [p[1], nearestGrid[1]], 'r')

    if gridLabels:
        for idx, pt in enumerate(grid.gridPts):
            f1.annotate(idx, pt, xycoords="data")

def sampleCircle(npts):
    pts = np.array([(np.cos(t), np.sin(t)) for t in np.arange(0, 2*np.pi, 2*np.pi/npts)])

    #shift to first quadrant
    pts += np.array([1,1])
    pts /= 2.0

    return pts

class Linktable(object):
    def __init__(self, grid, pts):
        self.grid = grid; self.pts = pts
        self.gridLookup  = [grid.nearestGridIdx(p) for p in pts]
        self.pointLookup = [[] for _ in range(grid.numPts)]
        for idx, p in enumerate(self.gridLookup):
            self.pointLookup[p].append(idx)
