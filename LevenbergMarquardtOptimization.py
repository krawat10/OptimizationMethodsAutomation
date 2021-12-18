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

    def numerical_differentiation(self, p):
        """ Numerical Differentiation
        Note: we are passing in the effor function for the model we are using, but
        we can substitute the error for the actual model function
            error(x + delta) - error(x) <==> f(x + delta) - f(x)
        :param p: values to be used in model
        :param args: input (x) and observations (y)
        :param error_function: function used to determine error based on params and observations
        :return: The jacobian for the error_function
        """
        delta_factor = 1e-3
        min_delta = 1e-4

        # Compute error
        y_0 = self.model.error_function([self.model.k0, p[0], p[1]])

        # Jacobian
        J = np.empty(shape=(len(p),) + y_0.shape, dtype=np.float)

        for i, param in enumerate(p):
            params_star = p[:]
            delta = param * delta_factor

            if abs(delta) < min_delta:
                delta = min_delta

            # Update single param and calculate error with updated value
            params_star[i] += delta
            y_1 = self.model.error_function([self.model.k0, params_star[0], params_star[1]])

            # Update Jacobian with gradients
            diff = y_0 - y_1
            J[i] = diff / delta

        return J

    def solve(self):
        points = [np.array([self.model.e0, self.model.w0])]
        llambda = 1e-2

        lambda_multiplier = 10
        kmax = 100

        # Equality : (JtJ + lambda * I * diag(JtJ)) * delta = Jt * error
        # Solve for delta
        params = [self.model.e0, self.model.w0]

        k = 0
        while k < kmax:
            k += 1
            points.append(np.array(params))
            # Retrieve jacobian of function gradients with respect to the params
            J = self.numerical_differentiation(params)
            Jt_x_J = inner(J, J)

            # I * diag(JtJ)
            A = eye(len(params)) * diag(Jt_x_J)

            # == Jt * error
            error = self.model.error_function([self.model.k0, params[0], params[1]])
            J_x_error = inner(J, error)

            rmserror = norm(error)

            print("{} RMS: {} Params: {}".format(k, rmserror, params))

            rmserror_star = rmserror + 1
            while rmserror_star >= rmserror:
                try:
                    delta = solve(Jt_x_J + llambda * A, J_x_error)
                except np.linalg.LinAlgError:
                    print("Error: Singular Matrix")
                    break

                # Update params and calculate new error
                params_star = params[:] + delta[:]
                error_star = self.model.error_function([self.model.k0, params_star[0], params_star[1]])
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
            # if reduction < 1e-18:
            if reduction < 1e-8:
                print("Change in error too small")
                break

        print('finish optimization (minimize)')
        print(f'Optimization for k:{self.model.k0:.8f}, e:{params[0]:.8f}, w:{params[1]:.8f}')
        self.plotter.plot([self.model.k0, params[0], params[1]], 'Levenberg Marquardt')
        self.plotter.e_w_scatter_area(points, 'Levenberg Marquardt', True)
