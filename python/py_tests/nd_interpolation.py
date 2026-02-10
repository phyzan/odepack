from odepack import ScatteredScalarField1D, ScatteredScalarField2D
import numpy as np


# 1D Example
print("=== 1D Example ===")
x = np.array([0, 5, 9, 3, 6, 4, 1])
y = 2*x+1

f = ScatteredScalarField1D(x, y)
print(f"f(3.5) = {f(3.5)}")  # Expected: 2*3.5 + 1 = 8


# 2D Example
print("\n=== 2D Example ===")
# Create scattered 2D points
points_2d = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, 0.5],
])

# Field values: f(x, y) = 3*x + 2*y + 1
values_2d = 3*points_2d[:, 0] + 2*points_2d[:, 1] + 1

f2d = ScatteredScalarField2D(points_2d, values_2d)

# Query at point (0.25, 0.25)
# Expected: 3*0.25 + 2*0.25 + 1 = 0.75 + 0.5 + 1 = 2.25
query = np.array([0.25, 0.25])
print(f"f(0.25, 0.25) = {f2d(0.25, 0.25)}")  # Expected: 2.25

# Query at point (0.5, 0.5)
# Expected: 3*0.5 + 2*0.5 + 1 = 1.5 + 1 + 1 = 3.5
query2 = np.array([0.5, 0.5])
print(f"f(0.5, 0.5) = {f2d(0.5, 0.5)}")  # Expected: 3.5
