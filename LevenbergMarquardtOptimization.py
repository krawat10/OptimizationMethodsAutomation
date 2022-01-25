import numpy as np
from numpy import inner, diag, eye
from numpy.linalg import norm, solve

from Model import Model
from Plotter import Plotter


class LevenbergMarquardtOptimization:
    def __init__(self, model: Model):
        self.model = model
        self.plotter = Plotter(model)

    # Not used function - for presentation purpose only
    def numerical_differentiation2(self, params):
        J = np.empty(shape=(len(params), self.model.size))
        J[0] = self.model.partial_derivative_e_2([self.model.k0, *params])
        J[1] = self.model.partial_derivative_w_2([self.model.k0, *params])

        return J

    def numerical_differentiation(self, params):
        delta_factor = 1e-4
        min_delta = 1e-4

        # Get response with given params
        y_0, dy_0, ddy_0 = self.model.integrate([self.model.k0, *params])

        # Initialize Jacobian Matrix
        J = np.empty(shape=(len(params),) + y_0.shape, dtype=np.float)

        for i, param in enumerate(params):
            new_params = params[:]
            delta = param * delta_factor

            if abs(delta) < min_delta:
                delta = min_delta

            # Update single param and calculate error with updated value
            new_params[i] += delta

            # Get response for f(x, B + delta)
            y_1, dy_1, ddy_1 = self.model.integrate([self.model.k0, *new_params])

            # Update Jacobian with partial derivatives
            J[i] = (y_1 - y_0) / delta

        return J

    def solve(self):
        points = []
        params = np.array([self.model.e0, self.model.w0])
        kmax = 10
        eps = 1e-18
        llambda = 10
        llambda_multiplier = 5

        for k in range(0, kmax):
            points.append(np.array(params))

            # Retrieve jacobian of function gradients with respect to the params
            J = self.numerical_differentiation(params)
            JtJ = inner(J, J)

            # I * diag(JtJ)
            A = eye(len(params)) * diag(JtJ)

            # == Jt * [y - f(B)]
            error = self.model.list_of_difference([self.model.k0, *params])
            Jerror = inner(J, error)
            norm_error = norm(error)
            norm_new_error = norm_error

            print(f'{k}, J: {self.model.quality_indicator([self.model.k0, *params])} RMS: {norm_error},  Params: {params}')
            while norm_new_error >= norm_error:
                delta = solve(JtJ + llambda * A, Jerror)

                # Update params and calculate new error
                new_params = params[:] + delta[:]
                norm_new_error = norm(self.model.list_of_difference([self.model.k0, *new_params]))

                if norm_new_error < norm_error:
                    params = new_params
                    llambda /= llambda_multiplier
                else:
                    llambda *= llambda_multiplier

                # Return if lambda explodes or if change is small
                if llambda > 1e9:
                    print("Lambda to large.")
                    break

            reduction = abs(norm_error - norm_new_error)
            if reduction < eps:
                print("Change in error too small")
                break

        print('finish optimization (minimize)')
        print(f'Optimization for k:{self.model.k0:.8f}, e:{params[0]:.8f}, w:{params[1]:.8f}')
        self.plotter.plot([self.model.k0, params[0], params[1]], 'Levenberg Marquardt')
        self.plotter.e_w_gradient_area(points, 'Levenberg Marquardt')
