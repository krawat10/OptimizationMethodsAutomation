import math

import numpy as np
import typing
from matplotlib import pyplot as plt
from scipy import misc

from Model import Model
from Plotter import Plotter


class GradientDescentOptimization:

    def __init__(self, model: Model):
        self.model = model
        self.plotter = Plotter(model)

    def solve(self):
        print('Gradient Descent Solve')
        # initial points
        e = self.model.e0
        w = self.model.w0
        z = self.model.quality_indicator([self.model.k0, e, w])

        points = [np.array([e, w])]

        alpha = 0.1  # learning rate
        max_iter = 40  # max iteration
        eps = 0.0001  # stop condition
        z0 = z

        for i in range(0, max_iter):
            # tmp_e = e - alpha * self.model.gain_func_partial_derivative(1, [self.model.k0, e, w])
            e = e - alpha * np.sum(self.model.partial_derivative_e([self.model.k0, e, w]))
            # tmp_w = w - alpha * (self.model.gain_func_partial_derivative(2, [self.model.k0, e, w]))
            w = w - alpha * np.sum(self.model.partial_derivative_w([self.model.k0, e, w]))

            z = self.model.quality_indicator([self.model.k0, e, w])

            cond = abs(z0 - z)
            z0 = z
            print(e, w, cond)
            points.append(np.array([e, w]))

            print(f'Gradient Descent iteration {i + 1}')
            if cond < eps:
                break

        self.plotter.e_w_gradient_area(points, 'Gradient Descent')
        self.plotter.plot([self.model.k0, e, w], 'Gradient Descent')

