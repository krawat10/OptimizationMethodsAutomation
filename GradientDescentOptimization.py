import math

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc

from Model import Model


class GradientDescentOptimization:

    def __init__(self, model):
        self.model = model

    def solve(self):
        print('GradientDescentOptimization')

        h = plt.contourf(self.model.XX, self.model.YY, self.model.ZZ.transpose())

        alpha = 0.1  # learning rate
        nb_max_iter = 100  # Nb max d'iteration
        eps = 0.0001  # stop condition

        x1_0 = self.model.e0  # start point
        x2_0 = self.model.w0
        z0 = self.model.gain_func_scalar([0.15894737, x1_0, x2_0])
        plt.scatter(x1_0, x2_0)

        cond = eps + 10.0  # start with cond greater than eps (assumption)
        nb_iter = 0
        tmp_z0 = z0
        while cond > eps and nb_iter < nb_max_iter:
            tmp_x1_0 = x1_0 - alpha * self.model.gain_func_partial_derivative(1, [0.15894737, x1_0, x2_0])
            tmp_x2_0 = x2_0 - alpha * self.model.gain_func_partial_derivative(2, [0.15894737, x1_0, x2_0])
            x1_0 = tmp_x1_0
            x2_0 = tmp_x2_0
            z0 = self.model.gain_func_scalar([0.15894737, x1_0, x2_0])
            nb_iter = nb_iter + 1
            cond = abs(tmp_z0 - z0)
            tmp_z0 = z0
            print(x1_0, x2_0, cond)
            plt.scatter(x1_0, x2_0)

        plt.title("GradientDescentOptimization")

        plt.savefig("gradiend_descent_2d_python.png", bbox_inches='tight')
        plt.show()
