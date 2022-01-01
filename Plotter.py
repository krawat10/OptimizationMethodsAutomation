import math

import numpy as np
import typing
from matplotlib import pyplot as plt
from scipy import misc

from Model import Model


class Plotter:
    def __init__(self, model: Model):
        self.model = model

    def plot(self, p, title):
        time = self.model.data['t'].values
        m, dm, ddm = self.model.integrate(p)
        plt.figure(1)
        plt.plot(time, m, 'r--', linewidth=1, label='m(t)')
        plt.plot(time, self.model.data['y'].values, 'b-', linewidth=1, label='y(t)')
        plt.plot(time, self.model.data['u'].values, 'g-', linewidth=1, label='u(t)')
        # plt.xlim([0, max(self.data['t'].head(400).values)])
        plt.ylim([-0.2, 0.2])
        plt.xlabel('Time')
        plt.ylabel('Response (y)')
        plt.title(f'{title} response for params k:{p[0]:.4f}, e:{p[1]:.4f}, w:{p[2]:.4f}')
        plt.savefig(f'images/plot-{title}.png')
        plt.show()

    def e_w_solve_area(self, title="Solve Area"):
        print(f'SolveArea E-W for k:{self.model.k0}')
        title += f' for k:{self.model.k0}'

        contours = plt.contour(self.model.EE, self.model.WW, self.model.EW.transpose(), 25)
        plt.clabel(contours, inline=True, fontsize=10)
        plt.title(title, fontsize=15)
        plt.xlabel('e', fontsize=11)
        plt.ylabel('w', fontsize=11)
        plt.colorbar()
        plt.legend(loc="upper right")
        plt.savefig(f'images/ew_gradient-{title}.png')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')  # Create the axes

        ax.plot_surface(self.model.EE, self.model.WW, self.model.EW.transpose())
        ax.set_xlabel('e')
        ax.set_ylabel('w')
        ax.set_zlabel('J')
        ax.set_title(title)

        ax.view_init(30, 40)
        plt.draw()

        plt.savefig(f'images/ew_solve-{title}.png')
        plt.show()

    def k_w_solve_area(self, title="Solve Area"):
        print(f'SolveArea K-W for e:{self.model.e0}')
        title += f' for e:{self.model.e0}'

        # Contour plot
        contours = plt.contour(self.model.EE, self.model.WW, self.model.EW.transpose(), 25)
        plt.clabel(contours, inline=True, fontsize=10)
        plt.title(title, fontsize=15)
        plt.xlabel('e', fontsize=11)
        plt.ylabel('w', fontsize=11)
        plt.colorbar()
        plt.legend(loc="upper right")
        plt.show()

        # 3d plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')  # Create the axes
        ax.plot_surface(self.model.KK, self.model.WW, self.model.KW.transpose())
        ax.set_xlabel('k')
        ax.set_ylabel('w')
        ax.set_zlabel('J')
        ax.set_title(title)
        plt.show()

    def e_w_scatter_area(self, points: np.array, title="Solve Area", connect=False):
        plt.contourf(self.model.EE, self.model.WW, self.model.EW.transpose())

        if connect:
            plt.plot(np.array(points).T[0], np.array(points).T[1], "x-")
        else:
            for point in points:
                plt.scatter(point[0], point[1])

        plt.xlabel('e')
        plt.ylabel('w')
        plt.title(title)
        plt.savefig(f'images/ew_scatter-{title}.png')
        plt.show()

    def e_w_gradient_area(self, points: np.array, title="Solve Area"):
        fig = plt.figure(figsize=(10, 7))
        contours = plt.contour(self.model.EE, self.model.WW, self.model.EW.transpose(), 25)
        plt.clabel(contours, inline=True, fontsize=10)
        plt.title(title, fontsize=15)
        if len(points) > 0:
            plt.plot(np.array(points).T[0], np.array(points).T[1])
            plt.plot(np.array(points).T[0], np.array(points).T[1], '*', label="Cost function")
        plt.xlabel('e', fontsize=11)
        plt.ylabel('w', fontsize=11)
        plt.colorbar()
        plt.legend(loc="upper right")
        plt.savefig(f'images/ew_gradient-{title}.png')
        plt.show()
