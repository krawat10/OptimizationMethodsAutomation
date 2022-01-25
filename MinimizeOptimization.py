import operator

import matplotlib.pyplot as plt
import numpy as np
import typing

from joblib import Parallel, delayed
from scipy.optimize import minimize
from cython.parallel import prange
from Model import Model
from Plotter import Plotter


class MinimizeOptimization:

    def __init__(self, model: Model):
        self.model = model
        self.plotter = Plotter(model)

    def minimalize_step(self, func) -> typing.List[int]:
        params = np.zeros((30 * 30 * 30, 3))

        idx = 0
        for k in np.linspace(0.1, 5, num=30):
            for e in np.linspace(0.1, 5, num=30):
                for w in np.linspace(0.1, 5, num=30):
                    params[idx][0] = k
                    params[idx][1] = e
                    params[idx][2] = w
                    idx += 1

        def process(k, e, w):
            return func([k, e, w]), k, e, w

        results = Parallel(n_jobs=6)(delayed(process)(param[0], param[1], param[2]) for param in params)

        J, min_k, min_e, min_w = min(results, key=lambda p: p[0])

        print(f'Optimalization for k:{min_k:.8f}, e:{min_e:.8f}, w:{min_w:.8f}')
        return [min_k, min_e, min_w]

    def solve(self):
        print('MinimizeOptimization')
        x0 = np.array([self.model.k0, self.model.e0, self.model.w0])

        self.model.integrate(x0)
        self.plotter.plot(x0, 'using odeint')

        # optimization
        # result = self.minimalize_step(self.model.quality_indicator)
        # print('finish optimization(step)')
        # print(f'Optimization for k:{result[0]:.8f}, e:{result[1]:.8f}, w:{result[2]:.8f}')
        # self.model.plot(result, 'Step')

        # optimization
        # result = minimize(self.model.quality_indicator, x0, method='Nelder-Mead')
        # self.model.k0, self.model.e0, self.model.w0 = result.x
        # print('finish optimization (minimize)')
        # print(f'Optimization for k:{self.model.k0:.8f}, e:{self.model.e0:.8f}, w:{self.model.w0:.8f}')
        # self.plotter.plot(result.x, 'Nelder-Mead')
