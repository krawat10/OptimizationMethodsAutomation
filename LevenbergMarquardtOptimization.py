import typing

from Model import Model
import numpy as np
from numpy import inner, max, diag, eye, Inf, dot
from numpy.linalg import norm, solve
import time

from Plotter import Plotter


class LevenbergMarquardtOptimization:
    def __init__(self, model: Model):
        self.model = model
        self.plotter = Plotter(model)

    def numerical_differentiation2(self, params):
        J = np.empty(shape=(len(params), self.model.size))
        J[0] = self.model.partial_derivative_e([self.model.k0, *params])
        J[1] = self.model.partial_derivative_w([self.model.k0, *params])

        return J

    def numerical_differentiation(self, params):
        delta_factor = 1e-3
        min_delta = 1e-3

        # Compute error
        y_0 = self.model.list_of_abs_difference([self.model.k0, *params])

        # Jacobian
        J = np.empty(shape=(len(params),) + y_0.shape, dtype=np.float)

        for i, param in enumerate(params):
            params_star = params[:]
            delta = param * delta_factor

            if abs(delta) < min_delta:
                delta = min_delta

            # Update single param and calculate error with updated value
            params_star[i] += delta
            y_1 = self.model.list_of_abs_difference([self.model.k0, *params_star])

            # Update Jacobian with gradients
            J[i] = (y_0 - y_1) / delta

        # J[0] = self.model.partial_derivative_e_2([self.model.k0, *params])
        # J[1] = self.model.partial_derivative_w_2([self.model.k0, *params])
        return J

    def solve(self):
        points = []
        params = np.array([self.model.e0, self.model.w0])
        kmax = 10
        llambda = 100
        lambda_multiplier = 10
        k = 0
        while k < kmax:
            k += 1
            points.append(np.array(params))
            # Retrieve jacobian of function gradients with respect to the params
            J = self.numerical_differentiation(params)
            JtJ = inner(J, J)

            # I * diag(JtJ)
            A = eye(len(params)) * diag(JtJ)

            # == Jt * error
            error = self.model.list_of_abs_difference([self.model.k0, *params])
            Jerror = inner(J, error)

            rmserror = norm(error)
            print("{} RMS: {} Params: {}".format(k, rmserror, params))

            rmserror_star = rmserror + 1
            while rmserror_star >= rmserror:
                try:
                    delta = solve(JtJ + llambda * A, Jerror)
                except np.linalg.LinAlgError:
                    print("Error: Singular Matrix")
                    return -1

                # Update params and calculate new error
                params_star = params[:] + delta[:]
                error_star = self.model.list_of_abs_difference([self.model.k0, *params_star])
                rmserror_star = norm(error_star)

                if rmserror_star < rmserror:
                    params = params_star
                    llambda /= lambda_multiplier
                    break

                llambda *= lambda_multiplier

                # Return if lambda explodes or if change is small
                if llambda > 1e9:
                    print("Lambda to large.")
                    break

            reduction = abs(rmserror - rmserror_star)
            if reduction < 1e-18:
                print("Change in error too small")
                break

        print('finish optimization (minimize)')
        print(f'Optimization for k:{self.model.k0:.8f}, e:{params[0]:.8f}, w:{params[1]:.8f}')
        self.plotter.plot([self.model.k0, params[0], params[1]], 'Levenberg Marquardt')
        self.plotter.e_w_scatter_area(points, 'Levenberg Marquardt', True)
