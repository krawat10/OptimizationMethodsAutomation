from pandas import read_csv
import numpy as np
from matplotlib.pyplot import xlim
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy import misc

from scipy.optimize import minimize, leastsq


class Model:
    k0 = 0.15894737
    e0 = 0.58386573
    w0 = 1.12294039

    XX = np.array([])
    YY = np.array([])
    ZZ = np.array([])

    def __init__(self, size=None):
        csv = read_csv('gr5.csv')

        if size is None:
            size = csv.shape[0]

        self.data = csv.head(size)
        self.size = size
        self.dy0 = (self.data['y'][0] - 0) / (self.data['t'][0])

    def mesh(self):
        xx = np.linspace(0.3, 0.6, 20)
        yy = np.linspace(1.1, 1.7, 20)
        self.ZZ = np.zeros((len(xx), len(yy)))
        self.XX, self.YY = np.meshgrid(xx, yy)

        for i in range(len(xx)):
            for j in range(len(yy)):
                self.ZZ[i][j] = self.gain_func_scalar([self.k0, xx[i], yy[j]])

    @staticmethod
    def float_round(num, places=1, direction=round):
        return direction(num * (10 ** places)) / float(10 ** places)

    def u(self, t):
        idx = self.data['t'].searchsorted(self.float_round(t, 1))
        try:
            u = self.data['u'][idx]
        except:
            u = self.data['u'][idx - 1]

        return u

    def model(self, solve, t, p):
        # Unpack parameters
        k, e, w = p
        u = self.u(t)

        # Unpack states
        x, dxdt = solve
        # Calculate derivatives as defined by the state equations ODEs
        # dx2dt = k * w ** 2 * u - (2 * e * w * dxdt + w ** 2 * x)
        # dx2dt = k * u - e * dxdt - w * x
        # dx2dt = (k * u - x - 2 * e * w * dxdt) / (w**2)

        dx2dt = (k * u) - (2 * e * w * dxdt) - (w ** 2 * x)  # best

        return [dxdt, dx2dt]

    def integrate(self, p):
        # print(f'Optimalization for k:{p[0]:.8f}, e:{p[1]:.8f}, w:{p[2]:.8f}')
        y = np.zeros(self.size)

        self.data['m'] = np.zeros(self.size)
        # initial y value
        y[0] = self.data['y'][0]

        # set initial y and dy
        x0 = [y[0], self.dy0]

        x = odeint(self.model, x0, self.data['t'].values, args=(p,))

        y = x[:, 0]
        self.data['m'] = x[:, 0]
        return y

    def gain_func_scalar(self, p):
        y = self.integrate(p)
        obj = 0
        for index, value in enumerate(y):
            obj += (value - self.data['y'][index]) ** 2
        return obj

    def gain_func_partial_derivative(self, var, p):
        args = p[:]

        def wraps(x):
            args[var] = x
            return self.gain_func_scalar(args)

        return misc.derivative(wraps, p[var], dx=1e-6)

    def gain_func_vec(self, ka, ea, wa):
        try:
            xy = np.zeros((len(ka), len(ea), len(wa)))
            for k in range(len(wa)):
                for j in range(len(ea)):
                    for i in range(len(ka)):
                        xy[i, j, k] = self.gain_func_scalar([ka[i], ea[i], wa[k]])
        except TypeError:
            xy = self.gain_func_scalar([ka, ea, wa])
        return xy
