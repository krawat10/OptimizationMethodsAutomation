from GradientDescentOptimization import GradientDescentOptimization
from MinimizeOptimization import MinimizeOptimization
from Model import Model
from SolveArea import SolveArea
from SteepestDescentOptimization import SteepestDescentOptimization

model = Model(1000)
model.mesh()

MinimizeOptimization(model).solve()
SolveArea(model).solve()
GradientDescentOptimization(model).solve()
SteepestDescentOptimization(model).solve()


