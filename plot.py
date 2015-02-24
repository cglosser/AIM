import matplotlib.pyplot as plt

def draw_points(grid, axis = None):
    if axis is None: 
        axis = plt.gca()
    x_coords, y_coords = grid.grid_points.transpose()
    axis.scatter(x_coords, y_coords, s=100, alpha=0.2)
    return axis

def draw_lines(grid, axis = None):
    if axis is None: 
        axis = plt.gca()
    x_coords, y_coords = grid.grid_points.transpose()
    axis.set_xticks(x_coords)
    axis.set_yticks(y_coords)
    axis.grid()
    return axis

def annotate_points(grid, axis = None):
    if axis is None:
        axis = plt.gca()
    for idx, point in enumerate(grid.grid_points): 
        axis.annotate(idx, point)
    return axis
        
def draw_object_and_grid(points, grid):
    fig    = plt.figure()
    subfig = fig.add_subplot(111)

    x_obj,  y_obj  = points.transpose()
    x_grid, y_grid = grid.grid_points.transpose()
    
    subfig.scatter(x_obj, y_obj, s=25, c='r', alpha=0.8)
    subfig.scatter(x_grid, y_grid, s=100, alpha=0.2)
