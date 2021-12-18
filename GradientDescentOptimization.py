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
        points = [np.array([self.model.e0, self.model.w0])]
        alpha = 0.1  # learning rate
        nb_max_iter = 100  # Nb max d'iteration
        eps = 0.0001  # stop condition

        e = self.model.e0  # start point
        w = self.model.w0
        z0 = self.model.quality_indicator([self.model.k0, e, w])

        points.append(np.array([e, w]))
        cond = eps + 10.0  # start with cond greater than eps (assumption)
        nb_iter = 0
        tmp_z0 = z0
        while cond > eps and nb_iter < nb_max_iter:
            tmp_e = e - alpha * self.model.gain_func_partial_derivative(1, [self.model.k0, e, w])
            tmp_w = w - alpha * self.model.gain_func_partial_derivative(2, [self.model.k0, e, w])

            e = tmp_e
            w = tmp_w

            z0 = self.model.quality_indicator([self.model.k0, e, w])
            nb_iter = nb_iter + 1
            cond = abs(tmp_z0 - z0)
            tmp_z0 = z0
            print(e, w, cond)
            points.append(np.array([e, w]))

        self.plotter.e_w_gradient_area(points, 'Gradient Descent')
        self.plotter.plot([self.model.k0, e, w], 'Gradient Descent')

