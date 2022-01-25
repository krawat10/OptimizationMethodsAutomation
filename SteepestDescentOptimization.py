import numpy as np
import scipy.optimize as optimize

from Model import Model
from Plotter import Plotter


class SteepestDescentOptimization:
    def __init__(self, model: Model):
        self.model = model
        self.plotter = Plotter(model)

    def solve(self):
        print('SteepestDescentOptimization')
        e = self.model.e0
        w = self.model.w0
        z0 = self.model.quality_indicator([self.model.k0, e, w])

        points = [np.array([self.model.e0, self.model.w0])]

        max_iter = 40  # max iteration
        eps = 0.0001  # stop condition

        for i in range(0, max_iter):
            partial_derivative_e = np.sum(self.model.gain_func_partial_derivative(1, [self.model.k0, e, w], 1e-1))
            partial_derivative_w = np.sum(self.model.gain_func_partial_derivative(2, [self.model.k0, e, w], 1e-1))

            best_alpha = optimize.golden(lambda alpha: self.model.quality_indicator([
                self.model.k0,
                e - alpha * partial_derivative_e,
                w - alpha * partial_derivative_w]))

            e = e - best_alpha * partial_derivative_e
            w = w - best_alpha * partial_derivative_w
            z = self.model.quality_indicator([self.model.k0, e, w])

            cond = abs(z0 - z)
            z0 = z
            print(e, w, cond)
            points.append(np.array([e, w]))

            print(f'Steepest Descent iteration {i + 1}')
            if cond < eps:
                break

        self.plotter.plot([self.model.k0, points[-1][0], points[-1][1]], 'Steepest Descent')

        print('finish optimization (minimize)')
        print(f'Optimization for J:{z0} k:{self.model.k0:.8f}, e:{points[-1][0]:.8f}, w:{points[-1][1]:.8f}')
        self.plotter.plot([self.model.k0, points[-1][0], points[-1][1]], f'Steepest Descent for k:{self.model.k0}')
        self.plotter.e_w_gradient_area(np.array(points), f'Steepest Descent for k:{self.model.k0}')
