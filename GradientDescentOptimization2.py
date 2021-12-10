import math

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc


class GradientDescentOptimization:
    def function(self, x1, x2):
        return - 1.0 * math.exp(-x1 ** 2 - x2 ** 2);  # todo - goal function here

    def partial_derivative(self, func, var=0, point=[]):
        args = point[:]

        def wraps(x):
            args[var] = x
            return func(*args)

        return misc.derivative(wraps, point[var], dx=1e-6)

    def solve(self):
        # ----------------------------------------------------------------------------------------#
        # Plot Function

        x1 = np.arange(-2.0, 2.0, 0.1)
        x2 = np.arange(-2.0, 2.0, 0.1)

        xx1, xx2 = np.meshgrid(x1, x2);

        z = - 1.0 * np.exp(-xx1 ** 2 - xx2 ** 2);

        h = plt.contourf(x1, x2, z)
        # plt.show()

        # ----------------------------------------------------------------------------------------#
        # Gradient Descent

        alpha = 0.1  # learning rate
        nb_max_iter = 100  # Nb max d'iteration
        eps = 0.0001  # stop condition

        x1_0 = 1.0  # start point
        x2_0 = 1.5
        z0 = self.function(x1_0, x2_0)
        plt.scatter(x1_0, x2_0)

        cond = eps + 10.0  # start with cond greater than eps (assumption)
        nb_iter = 0
        tmp_z0 = z0
        while cond > eps and nb_iter < nb_max_iter:
            tmp_x1_0 = x1_0 - alpha * self.partial_derivative(self.function, 0, [x1_0, x2_0])
            tmp_x2_0 = x2_0 - alpha * self.partial_derivative(self.function, 1, [x1_0, x2_0])
            x1_0 = tmp_x1_0
            x2_0 = tmp_x2_0
            z0 = self.function(x1_0, x2_0)
            nb_iter = nb_iter + 1
            cond = abs(tmp_z0 - z0)
            tmp_z0 = z0
            print(x1_0, x2_0, cond)
            plt.scatter(x1_0, x2_0)

        plt.title("GradientDescent Python (2d test)")

        plt.savefig("gradiend_descent_2d_python.png", bbox_inches='tight')
        plt.show()
