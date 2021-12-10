import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from Model import Model


class MinimizeOptimization:
    k = 0.15894737
    e = 0.58386573
    w = 1.12294039

    def __init__(self, model):
        self.model = model

    def plot(self, title):
        time = self.model.data['t'].values
        plt.figure(1)
        plt.plot(time, self.model.data['m'].values, 'r--', linewidth=1, label='m(t)')
        plt.plot(time, self.model.data['y'].values, 'b-', linewidth=1, label='y(t)')
        plt.plot(time, self.model.data['u'].values, 'g-', linewidth=1, label='u(t)')
        # plt.xlim([0, max(self.data['t'].head(400).values)])
        plt.ylim([-0.035, 0.035])
        plt.xlabel('Time')
        plt.ylabel('Response (y)')
        plt.title(f'{title} response for params k:{self.k:.4f}, e:{self.e:.4f}, w:{self.w:.4f}')
        plt.legend(loc='best')
        plt.savefig('2nd_order.png')
        plt.show()

    def minimalize_step(self, func):
        lowest_goal = 10000
        min_k = min_e = min_w = min_dy = 0

        for k in np.linspace(0.0, 0.2, num=10):
            for e in np.linspace(0.3, 0.9, num=10):
                for w in np.linspace(0.8, 1.2, num=10):
                    current_goal = func([k, e, w])
                    if current_goal < lowest_goal:
                        lowest_goal = current_goal
                        min_k = k
                        min_e = e
                        min_w = w
                    print(f'Optimalization for k:{k:.8f}, e:{e:.8f}, w:{w:.8f}')

        return min_k, min_e, min_w

    def solve(self):
        print('MinimizeOptimization')
        x0 = np.array([self.k, self.e, self.w])

        # initial params
        self.model.integrate(x0)
        self.plot('initial')

        # optimization
        # self.k, self.e, self.w = self.minimalize_step(self.model.objective)
        # print('finish optimization(step)')
        # print(f'Optimization for k:{self.k:.8f}, e:{self.e:.8f}, w:{self.w:.8f}')
        # self.plot('step')

        # optimization
        # result = minimize(self.model.objective_scalar, x0, method='BFGS')
        # self.k, self.e, self.w = result.x
        # print('finish optimization (minimize)')
        # print(f'Optimization for k:{self.k:.8f}, e:{self.e:.8f}, w:{self.w:.8f}')
        # self.plot('BFGS')
