import AIM
import numpy as np
from matplotlib.pylab import * 

#circ = AIM.sample_unit_circle(144)
#grid  = AIM.Grid(8)
#basis = AIM.build_basis_set(AIM.shift_to_first_quadrant(circ))

def main():
    circ = AIM.sample_unit_circle(128)
    grid = AIM.Grid(8)
    basis = AIM.build_basis_set(AIM.shift_to_first_quadrant(circ))

    m = AIM.construct_lambda(grid, basis)

    matshow(m.toarray())
    show()

if __name__ == "__main__":
    main()
