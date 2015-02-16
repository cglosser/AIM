import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from collections import namedtuple

BasisFunction = namedtuple("BasisFunction", ["start", "end", "mid"])

class GridNode(object):
    def __init__(self, num_cols, point):
        self.num_cols = num_cols
        self.point    = np.array(point).astype(int)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.num_cols != other.num_cols:
                raise Exception("Column mismatch in GridNode addition")
            return GridNode(self.num_cols, self.point + other.point)
        else:
            return GridNode(self.num_cols, 
                    self.point + np.array(other).astype(int))

    def __str__(self):
        return str(self.point)

    def to_index(self):
        return self.point[1]*self.num_cols + self.point[0]

class Grid(object):
    def __init__(self, grid_lines, size = 1):
        self.size = size
        self.grid_dim = grid_lines
        self.num_pts  = grid_lines**2
        self.grid_spacing = size/float(grid_lines - 1) #step size
        self.boxes = None
        self.grid_points = np.array([(x, y) for y in range(self.grid_dim)
            for x in range(self.grid_dim)])/float(self.grid_dim - 1)
        
    def __boundsCheck(self, points):
        """Check that all x,y pairs lie within [0, 1]. 
        Should make this an exception
        """
        assert np.all(points <= self.size) and np.all(points >= 0.0), \
                "point lies outside the grid"

    def anchor_node(self, point):
        """Compute the GridNode of the grid point to the south west of pt."""
        self.__boundsCheck(point)
        anchor = np.floor(point/self.grid_spacing).astype(int)
        return GridNode(self.grid_dim, anchor)
    
    def box_nodes(self, point):
        """Compute the four GridNodes of the box enclosing point"""
        anchor = self.anchor_node(point)
        return [anchor + [dx, dy] for dy in range(0, 2) for dx in range(0, 2)]

    def get_anchors(self, basis_funcs):
        """Build a list of anchor nodes corresponding to each basis function.

        The point at the south-west corner of each box uniquely indexes the
        box. This function builds a list of the appropriate GridNode for each
        basis function.
        """
        return [self.anchor_node(basis_func.mid)
                for basis_func in basis_funcs]

    def draw_points(self, axis = None):
        if axis is None: 
            axis = plt.gca()
        x_coords, y_coords = self.grid_points.transpose()
        axis.scatter(x_coords, y_coords, s=100, alpha=0.2)
        return axis
    
    def draw_lines(self, axis = None):
        if axis is None: 
            axis = plt.gca()
        x_coords, y_coords = self.grid_points.transpose()
        axis.set_xticks(x_coords)
        axis.set_yticks(y_coords)
        axis.grid()
        return axis

    def annotate_points(self, axis = None):
        if axis is None:
            axis = plt.gca()
        for idx, point in enumerate(self.grid_points): 
            axis.annotate(idx, point)
        return axis

        
def draw_object_and_grid(points, grid):
    fig    = plt.figure()
    subfig = fig.add_subplot(111)

    x_obj,  y_obj  = points.transpose()
    x_grid, y_grid = grid.grid_points.transpose()
    
    subfig.scatter(x_obj, y_obj, s=25, c='r', alpha=0.8)
    subfig.scatter(x_grid, y_grid, s=100, alpha=0.2)
    
def sample_circle(num_pts):
    pts = np.array([(np.cos(t), np.sin(t)) 
        for t in np.arange(0, 2*np.pi, 2*np.pi/num_pts)])

    #shift to first quadrant
    pts += np.array([1, 1])
    pts /= 2.0

    return pts

def sample_to_basis(pts):
    """Build a collection of rect (pulse) basis functions.
    
    Given a closed set of input points, convert each adjacent pair to a rect 
    basis function defined by start (given), end (given), and mid (average 
    of start and end) points.
    """
    return [BasisFunction(start=x1, end=x2, mid=(x1 + x2)/2.0) 
        for x1, x2 in zip(pts, np.roll(pts, -1, axis = 0))]

def q_matrix_element(m_vec, basis_func):
    if np.all(m_vec == 0):
        return 1 #integrating 1 from t = 0 to t = 1 -- easy analytic form
    func = lambda t: np.prod(np.power((1-t)*basis_func.start + 
        t*basis_func.end - basis_func.mid, m_vec))
    return quad(func, 0, 1)[0]

def w_matrix_element(m_vec, basis_func, u_vec):
    return np.prod(np.power(u_vec - basis_func.mid, m_vec))
