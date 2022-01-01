import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from Model import Model
from Plotter import Plotter


class SteepestDescentOptimization:
    def __init__(self, model: Model):
        self.model = model
        self.plotter = Plotter(model)

    def solve(self):
        print('SteepestDescentOptimization')
        guesses = [np.array([self.model.e0, self.model.w0])]
        # guesses = [np.array([1.25, 1])]
        #
        # Steepest Descent
        for i in range(0, 5):
            x = guesses[-1]
            s = np.array([
                -np.sum(self.model.partial_derivative_e([self.model.k0, x[0], x[1]])),
                -np.sum(self.model.partial_derivative_w([self.model.k0, x[0], x[1]]))
            ])

            def f1d(alpha):
                return self.model.quality_indicator([self.model.k0, *(x + alpha * s)])

            alpha_opt = optimize.golden(f1d)
            next_guess = x + alpha_opt * s
            guesses.append(next_guess)
            print(next_guess)
        #

        self.plotter.plot([self.model.k0, guesses[-1][0], guesses[-1][1]], 'Steepest Descent')

        print('finish optimization (minimize)')
        print(f'Optimization for k:{self.model.k0:.8f}, e:{guesses[-1][0]:.8f}, w:{guesses[-1][1]:.8f}')
        self.plotter.plot([self.model.k0, guesses[-1][0], guesses[-1][1]], 'Steepest Descent')
        self.plotter.e_w_scatter_area(np.array(guesses), 'Steepest Descent', True)

