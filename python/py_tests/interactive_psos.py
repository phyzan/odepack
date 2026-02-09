import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from odepack import RK45, PreciseEvent


# Perturbed Henon-Heiles system
def henon_heiles(t, q):
    """
    Perturbed Henon-Heiles system.
    q = [x, y, vx, vy]
    """
    x, y, vx, vy = q

    # Henon-Heiles potential derivatives
    dxdt = vx
    dydt = vy
    dvxdt = -x - 2*x*y
    dvydt = -y - x**2 + y**2

    return np.array([dxdt, dydt, dvxdt, dvydt])


# Define event: when y = 1
def event_condition(t, q):
    """Event triggers when y crosses 1"""
    return q[1] - 0.2


# Create the event
y_equals_1_event = PreciseEvent(
    name="y_equals_1",
    when=event_condition,
    direction=0,  # Detect crossing in any direction
    event_tol=1e-12, scalar_type="mpreal"
)

# Initial conditions: [x, y, vx, vy]
t0 = 0.0
q0 = np.array([0.0, 0.0, 0.3, 0.3])

# Create the RK45 solver with the event
solver = RK45(
    f=henon_heiles,
    t0=t0,
    q0=q0,
    rtol=1e-9,
    atol=1e-12,
    events=[y_equals_1_event], scalar_type="mpreal"
)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Henon-Heiles Trajectory (Click to advance)\nRed points: y = 1')
ax.grid(True, alpha=0.3)

# Storage for trajectory
trajectory_x = [q0[0]]
trajectory_y = [q0[1]]

# Plot elements
trajectory_line, = ax.plot([], [], 'b-', alpha=0.6, linewidth=1)
current_point = ax.scatter([q0[0]], [q0[1]], c='black', s=50, zorder=5)
event_points = ax.scatter([], [], c='red', s=50, zorder=6, marker='o')

# Storage for event points
event_x = []
event_y = []


def on_click(event):
    """Advance solver by one step when user clicks"""
    if event.button != MouseButton.LEFT:
        return

    if solver.is_dead:
        print("Solver has terminated. Cannot advance further.")
        return

    # Advance the solver by one step
    success = solver.advance()

    if not success:
        print("Failed to advance solver.")
        return

    # Get current state
    x, y, vx, vy = solver.q

    # Add to trajectory
    trajectory_x.append(x)
    trajectory_y.append(y)

    # Check if we're at an event
    if solver.event_located("y_equals_1"):
        event_x.append(x)
        event_y.append(y)
        print(f"Event detected at t={solver.t:.6f}, (x, y) = ({x:.6f}, {y:.6f})")
        # Update event points
        event_points.set_offsets(np.c_[event_x, event_y])

    # Update trajectory line
    trajectory_line.set_data(trajectory_x, trajectory_y)

    # Update current point
    current_point.set_offsets([[x, y]])

    # Update plot
    fig.canvas.draw_idle()

    # Print step info
    print(f"Step: t={solver.t:.6f}, (x, y) = ({x:.6f}, {y:.6f}), stepsize={solver.stepsize:.6e}")


# Connect the click event
cid = fig.canvas.mpl_connect('button_press_event', on_click)

print("Click on the plot to advance the solver by one step.")
print("Initial state: t={:.6f}, (x, y) = ({:.6f}, {:.6f})".format(t0, q0[0], q0[1]))

plt.show()


# USAGE: Run with `python test.py`. Left-click on the plot to advance the solver
# by one step. Current position is shown in black, events (y=0.2 crossings) in red.
