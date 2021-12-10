import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from Model import Model


class SteepestDescentOptimization:
    def __init__(self, model):
        self.model = model

    def solve(self):
        print('SteepestDescentOptimization')
        guesses = [np.array([self.model.e0, self.model.w0])]
        #
        # Steepest Descent
        for i in range(0, 5):
            x = guesses[-1]
            s = np.array([
                -self.model.gain_func_partial_derivative(1, [0.15894737, x[0], x[1]]),
                -self.model.gain_func_partial_derivative(2, [0.15894737, x[0], x[1]])
            ])

            #
            def f1d(alpha):
                p = [0.15894737, *(x + alpha * s)]
                return self.model.gain_func_scalar(p)

            #
            alpha_opt = optimize.golden(f1d)
            next_guess = x + alpha_opt * s
            guesses.append(next_guess)
            #
            print(next_guess)
        #

        plt.contour(self.model.XX, self.model.YY, self.model.ZZ.transpose(), 50)
        it_array = np.array(guesses)
        plt.plot(it_array.T[0], it_array.T[1], "x-")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('SteepestDescentOptimization')
        plt.show()
