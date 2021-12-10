import math

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc

from Model import Model


class SolveArea:
    def __init__(self, model):
        self.model = model

    def solve(self):
        print('SolveArea')

        plt.contourf(self.model.XX, self.model.YY, self.model.ZZ.transpose())
        plt.xlabel('e')
        plt.ylabel('w')
        plt.title("Solve Area")
        plt.show()
