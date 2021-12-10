import numpy as np
import numpy.linalg as la

import scipy.optimize as optimize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


class SteepestDescentOptimization:
    def f(self, x):
        return 0.5 * x[0] ** 2 + 2.5 * x[1] ** 2

    def df(self, x):
        return np.array([x[0], 5 * x[1]])

    def solve(self):
        xmesh, ymesh = np.mgrid[-2:2:50j, -2:2:50j]
        fmesh = self.f(np.array([xmesh, ymesh]))

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(xmesh, ymesh, fmesh)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        fig.show()

        guesses = [np.array([2, 1.0])]
        #
        # Steepest Descent
        for i in range(0, 5):
            x = guesses[-1]
            s = -self.df(x)
            #
            def f1d(alpha):
                return self.f(x + alpha * s)
            #
            alpha_opt = optimize.golden(f1d)
            next_guess = x + alpha_opt * s
            guesses.append(next_guess)
            #
            print(next_guess)
        #

        plt.contour(xmesh, ymesh, fmesh, 50)
        it_array = np.array(guesses)
        plt.plot(it_array.T[0], it_array.T[1], "x-")
        plt.show()
