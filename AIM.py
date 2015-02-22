import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import hankel2
from collections import namedtuple

BasisFunction = namedtuple("BasisFunction", ["start", "end", "mid"])


class Grid(object):
    Node = namedtuple("Node", ["location","index"])

    def __init__(self, grid_dim, size = 1):
        self.size         = size
        self.grid_dim     = grid_dim
        self.grid_spacing = size/float(grid_dim - 1)

        self.nodes = [Grid.Node(np.array([x, y]), self.__node_index([x, y]))
                for y in range(grid_dim) for x in range(grid_dim)]

        self.grid_points = np.array([node.location/float(grid_dim - 1)
                for node in self.nodes])

    def __node_index(self, location):
        """Convert an integral (r, c) grid coordinate to its unique index."""
        return location[0] + self.grid_dim*location[1]
        
    def anchor(self, point):
        """Compute the Node of the grid point to the south west of point"""
        anchor = np.floor(point/self.grid_spacing).astype(int)
        return self.nodes[self.__node_index(anchor)]
    
    def box_nodes(self, point):
        """Compute the four Nodes of the box enclosing point"""
        anchor = self.anchor(point)
        indices = [self.nodes[self.__node_index(anchor.location + 
            np.array([dx, dy]))] for dy in range(0, 2) for dx in range(0, 2)]
        return indices

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

def green_2d(k, rho1, rho2):
    return hankel2(0, k*np.norm(rho1 - rho2))/4j
        
def draw_object_and_grid(points, grid):
    fig    = plt.figure()
    subfig = fig.add_subplot(111)

    x_obj,  y_obj  = points.transpose()
    x_grid, y_grid = grid.grid_points.transpose()
    
    subfig.scatter(x_obj, y_obj, s=25, c='r', alpha=0.8)
    subfig.scatter(x_grid, y_grid, s=100, alpha=0.2)

def shift_to_first_quadrant(points):
    x_bounds = np.array([np.min(points[:, 0]), np.max(points[:, 0])])
    y_bounds = np.array([np.min(points[:, 1]), np.max(points[:, 1])])
    x_span, y_span = x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0]

    scale_factor = np.max([x_span, y_span])
    return (points - np.array([x_bounds[0], y_bounds[0]]))/scale_factor
    
def sample_unit_circle(num_pts):
    """ Sample a circle of diameter = 1 in the first quadrant."""
    return np.array([(np.cos(t), np.sin(t)) 
        for t in np.arange(0, 2*np.pi, 2*np.pi/num_pts)])

def build_basis_set(pts):
    """Build a collection of rect (pulse) basis functions.
    
    Given a closed set of input points, convert each adjacent pair to a rect 
    basis function defined by start (given), end (given), and mid (average 
    of start and end) points.
    """
    return [BasisFunction(start=x1, end=x2, mid=(x1 + x2)/2.0) 
        for x1, x2 in zip(pts, np.roll(pts, -1, axis = 0))]

def rhs_q_matrix_element(m_vec, basis_func):
    if np.all(m_vec == 0):
        return 1 #integrating 1 from t = 0 to t = 1 -- easy analytic form
    func = lambda t: np.prod(np.power((1-t)*basis_func.start + 
        t*basis_func.end - basis_func.mid, m_vec))
    return quad(func, 0, 1)[0]

def lhs_w_matrix_element(m_vec, u_vec, expansion_point):
    return np.prod(np.power(u_vec - expansion_point, m_vec))

def find_grid_mapping(grid, basis_func):
    box_nodes = grid.box_nodes(basis_func.mid)
    combo_m = [(mx, my) for my in range(2) for mx in range(2)]

    lhs = np.array([[lhs_w_matrix_element(m_vec, basis_func.mid,
        corner.location) for m_vec in combo_m] 
        for corner in box_nodes])

    rhs = np.array([rhs_q_matrix_element(m_vec, basis_func) 
        for m_vec in combo_m])

    return np.linalg.solve(lhs, rhs)
