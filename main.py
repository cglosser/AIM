import AIM
import numpy as np
import timeit
from matplotlib.pylab import * 

NUM_BASIS = 32
NUM_GRID  = 12

def naiive_trials(num_trials = 5):
    num_basis = 2**np.arange(2,7)
    for n in num_basis:
        pts   = AIM.shift_to_first_quadrant(AIM.sample_unit_circle(n))
        basis = AIM.build_basis_set(pts)

        time_average = 0.0
        for trial in range(num_trials):
           r_vec = np.random.rand(n)
           tick = timeit.default_timer()
           result = AIM.naiive_interaction_matrix(AIM.green_2d, basis).dot(r_vec)
           tock = timeit.default_timer()
           time_average += tock - tick
        
        print n, time_average/num_trials
    return

def AIM_trials(grid_dim, num_trials = 5):
    num_basis = np.arange(5,100,5)
    grid = AIM.Grid(grid_dim)

    for n in num_basis:
        pts   = AIM.shift_to_first_quadrant(AIM.sample_unit_circle(n))
        basis = AIM.build_basis_set(pts)

        time_average = 0.0
        for trial in range(num_trials):
            r_vec = np.random.rand(n)

            tick = timeit.default_timer()
            lam = AIM.construct_lambda(grid, basis)
            green_mat = AIM.green_matrix(AIM.green_2d, grid)
            result = lam.dot(green_mat).dot(lam.transpose().toarray()).dot(r_vec)
            tock = timeit.default_timer()
            time_average += tock - tick
        
        print n, time_average/num_trials
    return

def main():
    pts   = AIM.sample_unit_circle(NUM_BASIS)
    basis = AIM.build_basis_set(AIM.shift_to_first_quadrant(pts))
    grid  = AIM.Grid(12)

    r_vec = np.random.rand(NUM_BASIS)

    m1 = AIM.naiive_interaction_matrix(AIM.green_2d, basis).dot(r_vec)
    lam = AIM.construct_lambda(grid, basis)
    m2 = lam.dot(AIM.green_matrix(AIM.green_2d, grid)).dot(lam.transpose().toarray()).dot(r_vec)

    print m1
    print m2

    print "=="*50
    print m2 - m1
    print "RMS:", np.sqrt(np.sum(np.abs(m2-m1))/NUM_BASIS)

    #grid  = AIM.Grid(NUM_GRID)

    #print "Testing AIM implementation..."
    #AIM_trials(12)

    #print
    #print "Testing naiive implementation..."
    #naiive_trials()

if __name__ == "__main__":
    main()
