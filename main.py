import AIM
import numpy as np
from matplotlib.pylab import * 

def sample_wiggle_circle(num_pts, wiggle = 0):
    angles = np.arange(0, 2*np.pi, 2*np.pi/num_pts)
    return np.array([np.array([np.cos(theta), np.sin(theta)])*(1 + 
        np.cos(wiggle*theta)/5) for theta in angles])

def main():
    #circ = sample_wiggle_circle(512, 10)
    circ = AIM.sample_unit_circle(128)
    grid = AIM.Grid(8)
    basis = AIM.build_basis_set(AIM.shift_to_first_quadrant(circ))

    #lambda_src = AIM.construct_lambda(grid, basis)
    g = grid.green_matrix(AIM.green_2d)
    #m = lambda_src.toarray().dot(g)
    #lp = lambda_src.transpose().toarray()

    #m2 = np.dot(m, lp)
    for i in g[0:8,0]:
        print i

    matshow(abs(g))
    show()


if __name__ == "__main__":
    main()
