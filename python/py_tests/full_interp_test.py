from odepack import *
import matplotlib.pyplot as plt


"""
Testing that the constructors of interpolators, scalar fields and vector fields work properly
"""


"""
RegularGridInterpolator
"""

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 200)

X, Y = np.meshgrid(x, y, indexing='ij')

F = X+Y
G = X-Y
H = X*Y
J = X/Y

#gather all and reshape to (nx, ny, 2, 2)
R = np.stack([F, G, H, J], axis=-1).reshape(X.shape + (2, 2))
# now R[xi, xj] is a 2x2 matrix containing the values of [[F, G], [H, J]] at that grid point

#1D case:
f = RegularGridInterpolator(x*x, x)
print("\nRegularGridInterpolator 1D: expected approximately 9")
print(f(3), " expected approximately 9")

#2D case:
f = RegularGridInterpolator(R, x, y)

print("\nRegularGridInterpolator 2D: expected [[10, -4], [21, 0.43]]")
print(f(3, 7))
"""
expected
[[10, -4],
[21, 0.43]]
"""



"""
ScatteredNdInterpolator
"""

# flatten points to resemble scattered data
points = np.column_stack((X.ravel(), Y.ravel()))

#we need a values shape of (nx*ny, 2, 2)
values = np.column_stack((F.ravel(), G.ravel(), H.ravel(), J.ravel())).reshape(-1, 2, 2)

#1D case:
f = ScatteredNdInterpolator(points=x, values=x*x)
print("\nScatteredNdInterpolator 1D (points): expected approximately 9")
print(f(3), " expected approximately 9")

tri_1d = DelaunayTri(points=x)
print(tri_1d.total_volume, " should be approximately 20, the length of the bounding box")
g = ScatteredNdInterpolator(tri=tri_1d, values=x*x)
print("\nScatteredNdInterpolator 1D (tri): expected approximately 9")
print(g(3), " expected approximately 9")

f = ScatteredNdInterpolator(points=points, values=values)
print("\nScatteredNdInterpolator 2D (points): expected [[10, -4], [21, 0.43]]")
print(f(3, 7))

tri = DelaunayTri(points=points)
print(tri.total_volume, " should be approximately 400, the area of the bounding box")
quit()
g = ScatteredNdInterpolator(tri=tri, values=values)
print("\nScatteredNdInterpolator 2D (tri): expected [[10, -4], [21, 0.43]]")
print(g(3, 7))
"""
expected
[[10, -4],
[21, 0.43]]
"""


"""
TESTING SCALAR FIELDS
"""

f = RegularGridScalarField(F, x, y)

print("\nRegularGridScalarField: expected approximately 13")
print(f(3, 10), " expected approximately 13")

try:
    f(11, 8) # this should raise error
except ValueError as e:
    print(e)



f = ScatteredScalarField(points=points, values=F.ravel())

print("\nScatteredScalarField (points): expected approximately 13")
print(f(3, 10), " expected approximately 13")

tri = DelaunayTri(points=points)
g = ScatteredScalarField(tri=tri, values=F.ravel())
print("\nScatteredScalarField (tri): expected approximately 13")
print(g(3, 10), " expected approximately 13")

try:
    f(11, 8) # this should raise error
except ValueError as e:
    print(e)


"""
TESTING VECTOR FIELDS
"""

f = RegularGridVectorField([F, G], x, y)
print("\nRegularGridVectorField: expected approximately [13, -7]")
print(f(3, 10), " expected approximately [13, -7]")
try:
    f(11, 8) # this should raise error
except ValueError as e:
    print(e)

f = ScatteredVectorField(points=points, values=[F.ravel(), G.ravel()])
print("\nScatteredVectorField (points): expected approximately [13, -7]")
print(f(3, 10), " expected approximately [13, -7]")

tri = DelaunayTri(points=points)
g = ScatteredVectorField(tri=tri, values=[F.ravel(), G.ravel()])
print("\nScatteredVectorField (tri): expected approximately [13, -7]")
print(g(3, 10), " expected approximately [13, -7]")
try:
    f(11, 8) # this should raise error
except ValueError as e:
    print(e)
# now with ScatteredVectorField


# Now perform a streamplot

print("\nSTREAMPLOT TEST")

f_rg = RegularGridVectorField([F, G], x, y)
lines_rg = f_rg.streamplot_data(max_length=40, ds=0.1, density=30)
fig, ax = plt.subplots()
for line in lines_rg:
    ax.plot(line[0], line[1])
ax.set_title("RegularGridVectorField streamplot")
plt.show()

