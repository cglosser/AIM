import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg    import norm
from scipy.integrate import quad
from scipy.sparse    import dok_matrix, coo_matrix
from scipy.special   import hankel2
from collections     import namedtuple

BasisFunction = namedtuple("BasisFunction", ["start", "end", "mid"])

class Grid(object):
    Node = namedtuple("Node", ["location","index"])

    class BoundaryError(Exception):
        def __init__(self, point):
            self.point = point
            self.msg   = "grid point " + str(point)

        def __str__(self):
            return self.msg

    def __init__(self, grid_dim):
        self.grid_dim     = grid_dim
        self.num_nodes    = grid_dim**2
        self.grid_spacing = 1/float(grid_dim - 1)

        self.nodes = [Grid.Node(np.array([x, y]), self.__node_index([x, y]))
                for y in range(grid_dim) for x in range(grid_dim)]

        self.grid_points = np.array([self.absolute_location(node.location)
                for node in self.nodes])

    def __node_index(self, location):
        """Convert an integral (x, y) grid coordinate to its unique index."""
        if any([i < 0 or i >= self.grid_dim for i in location]):
            raise Grid.BoundaryError(location)
        return location[0] + self.grid_dim*location[1]

    def absolute_location(self, grid_coord):
        """Give the absolute box location of integral grid coordinates."""
        return grid_coord/float(self.grid_dim - 1)
        
    def anchor(self, pos):
        """Return the Node corresponding to the nearest grid 
        point south west of pos.
        """
        anchor = np.floor(pos/self.grid_spacing).astype(int)
        return self.nodes[self.__node_index(anchor)]
    
    def box_nodes(self, point, degree = 0):
        """Compute the indices of Nodes enclosing point. The degree defines
        the "box radius" for the indices (degree == 0 means the four nearest
        corners, degree == 1 gives the next twelve points and so on).
        """
        delta_range = range(-degree, degree + 2)
        anchor_loc = self.anchor(point).location
        indices = [self.nodes[self.__node_index(anchor_loc + 
            np.array([dx, dy]))] for dy in delta_range for dx in delta_range]
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
    
    def green_matrix(self, green_function):
        g_mat = np.zeros([self.num_nodes, self.num_nodes], dtype=complex)
        for node1 in self.nodes:
            for node2 in self.nodes:
                if node1 is node2:
                    g_mat[node1.index, node2.index] = 1
                else:
                    pos1 = self.absolute_location(node1.location)
                    pos2 = self.absolute_location(node2.location)
                    g_mat[node1.index, node2.index] = green_function(pos2, pos1)

        return g_mat

def green_2d(vec_1, vec_2, k = 1):
    return hankel2(0, k*norm(vec_1 - vec_2))/4j
        
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
    angles = np.arange(0, 2*np.pi, 2*np.pi/num_pts)
    return np.array([(np.cos(theta), np.sin(theta)) for theta in angles])

def build_basis_set(pts):
    """Build a collection of rect (pulse) basis functions.
    
    Given a closed set of input points, convert each adjacent pair to a rect 
    basis function defined by start (given), end (given), and mid (average 
    of start and end) points.
    """
    point_pairs = zip(pts, np.roll(pts, -1, axis = 0))
    return [BasisFunction(start=x1, end=x2, mid=(x1 + x2)/2.0) 
        for x1, x2 in point_pairs]

def rhs_q_matrix_element(m_vec, basis_func):
    func = lambda t: np.prod(np.power((1-t)*basis_func.start + 
        t*basis_func.end - basis_func.mid, m_vec))
    return quad(func, 0, 1)[0]

def lhs_w_matrix_element(m_vec, u_vec, expansion_point):
    return np.prod(np.power(u_vec - expansion_point, m_vec))

def find_grid_mapping(grid, basis_func, degree = 0):
    point_range = range(degree + 2)
    combo_m = [(mx, my) for my in point_range for mx in point_range]

    box_nodes = grid.box_nodes(basis_func.mid, degree)

    abs_loc = grid.absolute_location
    lhs = np.array([
        [lhs_w_matrix_element(m_pair, basis_func.mid, abs_loc(node.location)) 
            for node in box_nodes]
            for m_pair in combo_m 
    ])

    rhs = np.array([rhs_q_matrix_element(m_vec, basis_func) 
        for m_vec in combo_m])

    indices = [node.index for node in box_nodes]
    solution = np.linalg.solve(lhs, rhs)

    return zip(indices, solution)

def construct_lambda(grid, basis_funcs, degree = 0):
    num_basis_funcs = len(basis_funcs)
    num_grid_points = grid.num_nodes

    lambda_matrix = dok_matrix((num_basis_funcs, num_grid_points))
    for row, pulse in enumerate(basis_funcs):
        for col, projection in find_grid_mapping(grid, pulse, degree):
            print row, col, projection
            lambda_matrix[row, col] = projection

    return lambda_matrix.asformat("coo")
