from odepack import *
from numiphy.findiffs import *
import matplotlib.pyplot as plt


t, x, y, px, py = symbols('t, x, y, px, py')

# define potential in 2D and sample it on a grid

V_exact = (x**2 + y**2) / 2 + x**2*y - y**3 / 3

grid = Uniform1D(-30, 30, 1000) * Uniform1D(-40, 40, 1000)

X, Y = np.meshgrid(*grid.x, indexing="ij")

V_array = (X**2 + Y**2) / 2 + X**2*Y - Y**3 / 3

#define the sampled potential as a symbolic object with built-in interpolation and finite differences
V = RegularScalarField("V", grid, V_array, x, y)

Vx = V.diff(x)
Vy = V.diff(y)

#############
## Create unstructured grid and test interpolation on it
#############
Vx = Vx.as_unstructured()
Vy = Vy.as_unstructured()

print("Creating interpolators")
Vx.to_sampled_scalar_field
Vy.to_sampled_scalar_field
print("Done")

#define equations of motion symbolically
qdot = [px, py, -Vx(x, y), -Vy(x, y)]

#define the system using the exact potential for comparison
qdot_exact = [px, py, -V_exact.diff(x), -V_exact.diff(y)]

#define the system of ODEs symbolically
ode_system = OdeSystem(qdot, t, [x, y, px, py])

# define an orbit using the ode_system object
x0, y0, px0, py0 = 0, 0, 0.3, 0.3

print("getting orbit")
orbit = ode_system.get(t0=0, q0=[x0, y0, px0, py0], rtol=1e-9, atol=1e-12, method="RK45")
print("done")

#integrate
result = orbit.integrate(10, max_prints=100000)
result.examine()

plt.plot(*result.q.T[:2])
plt.grid()

plt.show()