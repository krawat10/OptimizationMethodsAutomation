from os.path import isfile

import numpy as np
from numpy import savez_compressed, load
from pandas import read_csv
from scipy import misc
from scipy.integrate import odeint


class Model:
    k0 = 0.15894737
    e0 = 0.58386573
    w0 = 1.12294039

    KK = np.array([])
    EE = np.array([])
    WW = np.array([])
    EW = np.array([])
    KW = np.array([])

    def __init__(self):
        csv = read_csv('gr5.csv')

        self.data = csv.tail(4500).reset_index(drop=True)
        self.size: int = self.data.shape[0]
        self.dy0 = (self.data['y'][0] - 0) / (self.data['t'][0])

    def generate_mesh(self):
        kk1 = np.linspace(0.1, 1.0, 15)
        ee1 = np.linspace(0.10, 0.65, 15)
        ww1 = np.linspace(1.0, 1.6, 15)

        self.EW = np.zeros((len(ee1), len(ww1)))
        self.KW = np.zeros((len(kk1), len(ww1)))
        self.EE, self.WW = np.meshgrid(ee1, ww1)
        self.KK, self.WW = np.meshgrid(kk1, ww1)

        if isfile('ew4.npz'):
            self.EW = load('ew4.npz')['arr_0']
        else:
            for i in range(len(ee1)):
                for j in range(len(ww1)):
                    self.EW[i][j] = self.quality_indicator([self.k0, ee1[i], ww1[j]])

        savez_compressed('ew4', self.EW)

        if isfile('kw4.npz'):
            self.KW = load('kw4.npz')['arr_0']
        else:
            for i in range(len(kk1)):
                for j in range(len(ww1)):
                    self.KW[i][j] = self.quality_indicator([kk1[i], self.e0, ww1[j]])

        savez_compressed('kw4', self.KW)

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
        x, dxdt = solve  # vyk, xyk, vr

        # Calculate derivatives as defined by the state equations ODEs
        dx2dt = (k * u) - (2 * e * w * dxdt) - (w ** 2 * x)  # best
        return [dxdt, dx2dt]

    def integrate(self, p):
        y = np.zeros(self.size)
        self.data['m'] = np.zeros(self.size)

        # initial y value
        y[0] = self.data['y'][0]

        # set initial y and dy
        x0 = [y[0], 0]
        result = odeint(self.model, x0, self.data['t'].values, args=(p,))

        self.data['m'] = result[:, 0]
        self.data['dm'] = result[:, 1]
        y = result[:, 0]
        dy = result[:, 1]
        return [y, dy]

    def quality_indicator(self, p):
        y, dy = self.integrate(p)

        indicator = np.sum(np.power((y - self.data['y']), 2))
        print(f'J: {indicator} for k: {p[0]}, e:{p[1]}, w:{p[2]}')
        return indicator

    def list_of_difference(self, p):
        y, dy, ddy = self.integrate(p)

        return self.data['y'] - y

    def gain_func_partial_derivative(self, var, p, dx=1e-1):
        args = p[:]

        def wraps(x):
            args[var] = x
            return self.quality_indicator(args)

        # analitycs (manual)
        derivative = misc.derivative(wraps, p[var], dx=dx)
        return derivative
