

from GradientDescentOptimization import GradientDescentOptimization
from LevenbergMarquardtOptimization import LevenbergMarquardtOptimization
from MinimizeOptimization import MinimizeOptimization
from Model import Model
from Plotter import Plotter
from SteepestDescentOptimization import SteepestDescentOptimization

model = Model()
model.generate_mesh()
# MinimizeOptimization(model).solve()
Plotter(model).e_w_solve_area()
Plotter(model).e_w_gradient_area([], f'Solve area for k={model.k0}')
Plotter(model).k_w_solve_area()
#

GradientDescentOptimization(model).solve()
SteepestDescentOptimization(model).solve()
LevenbergMarquardtOptimization(model).solve()




#





