import multiprocessing
from os.path import isfile

from joblib import Parallel, delayed
from numpy import savez_compressed, load
from pandas import read_csv
import numpy as np
from matplotlib.pyplot import xlim
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy import misc
from scipy import signal

from scipy.optimize import minimize, leastsq


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

        # if size is None:
        #     size = csv.shape[0]

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
        k, e, w = p
        self.data['m'] = np.zeros(self.size)
        # initial y value
        y[0] = self.data['y'][0]

        # set initial y and dy
        x0 = [y[0], 0]

        # self.model -> dx2dt = (k * u) - (2 * e * w * dxdt) - (w ** 2 * x)
        x = odeint(self.model, x0, self.data['t'].values, args=(p,))

        self.data['m'] = x[:, 0]
        self.data['dm'] = x[:, 1]
        self.data['ddm'] = (k * self.data['u']) - (2 * e * w * self.data['dm']) - (w ** 2 * self.data['m'])
        y = x[:, 0]
        dy = x[:, 1]
        ddy = self.data['ddm']
        return [y, dy, ddy]

    def quality_indicator(self, p):
        indicator = np.sum(self.list_of_square_difference(p))
        print(f'J: {indicator} for k: {p[0]}, e:{p[1]}, w:{p[2]}')
        return indicator

    def abs_quality_indicator(self, p):
        indicator = np.sum(self.list_of_abs_difference(p))
        print(f'J: {indicator} for k: {p[0]}, e:{p[1]}, w:{p[2]}')
        return indicator

    def list_of_square_difference(self, p):
        y, dy, ddy = self.integrate(p)

        return np.power((y - self.data['y']), 2)

    def list_of_abs_difference(self, p):
        y, dy, ddy = self.integrate(p)

        return np.abs((y - self.data['y']))

    def gain_func_partial_derivative(self, var, p, dx=1e-1):
        args = p[:]

        def wraps(x):
            args[var] = x
            return self.quality_indicator(args)

        # analitycs (manual)
        derivative = misc.derivative(wraps, p[var], dx=dx)
        return derivative

    def partial_derivative_e(self, p):
        k, e, w = p
        m, dmdt, d2mdt2 = self.integrate(p)

        dmde = -(2 * self.data['dm']) / w

        # ((y - m)^2)'
        # (y^2 - 2ym + m^2)'
        # -2ym' + (m^2)'
        # -2ym' + 2mm'
        square_partial = -2 * self.data['y'] * dmde + 2 * m * dmde

        return square_partial

    def partial_derivative_w(self, p):
        k, e, w = p
        m, dmdt, d2mdt2 = self.integrate(p)
        u = self.data['u']

        dmdw = (-2 * k * u) / (w ** 3) + (2 * d2mdt2) / (w ** 3) + (2 * e * dmdt) / (w ** 2)
        # ((y - m)^2)'
        # (y^2 - 2ym + m^2)'
        # (-2ym + m^2)'
        # -2ym' + (m^2)'
        # -2ym' + 2mm'
        square_partial = -2 * self.data['y'] * dmdw + 2 * m * dmdw

        return square_partial

    def partial_derivative_w_2(self, p):
        k, e, w = p
        m, dmdt, d2mdt2 = self.integrate(p)

        def w_partial_model(solve, w, params):
            k, e, idx = params

            # Unpack states
            y, dydw = solve
            v = self.data['dm'][idx]
            x = self.data['m'][idx]

            # Calculate derivatives
            dy2dw = -2 * e * w * dydw - 2 * e * v - w ** 2 * y - 2 * w * x  # best
            return [dydw, dy2dw]

        dmdw = np.zeros(self.size)

        for i in range(0, self.size):
            params = [k, e, i]
            arr = odeint(w_partial_model, [0, 0], [0, 0.1, w - 0.1, w], args=(params,))

            dmdw[i] = arr[-1, 0]

        square_partial = -2 * self.data['y'] * dmdw + 2 * m * dmdw

        return square_partial

    def partial_derivative_e_2(self, p):
        k, e, w = p
        m, dmdt, d2mdt2 = self.integrate(p)

        def e_partial_model(solve, e, params):
            k, w, idx = params

            # Unpack states
            y, dyde = solve  # vyk, xyk, vr

            v = self.data['dm'][idx]
            x = self.data['m'][idx]

            # Calculate derivatives as defined by the state equations ODEs
            dy2de = -2 * e * w * dyde - 2 * w * v - w ** 2 * y  # best
            return [dyde, dy2de]

        dmde = np.zeros(self.size)

        for i in range(0, self.size):
            params = [k, w, i]
            arr = odeint(e_partial_model, [0, 0], [0, 0.1, e - 0.1, e], args=(params,))
            dmde[i] = arr[-1, 0]

        return -2 * self.data['y'] * dmde + 2 * m * dmde
