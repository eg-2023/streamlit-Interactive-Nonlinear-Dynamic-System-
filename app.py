#!/usr/bin/env python3
"""
Streamlit GUI for Interactive Simulation and Visualization of Chaotic Nonlinear Behavior of a Double Pendulum Using Python:
- Animation (GIF via Pillow)
- Angle–time plot
Run: streamlit run app.py
"""
import os
import io
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for Streamlit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import odeint
import streamlit as st

FIXED_FPS = 50  # fixed animation speed

# ---------------------------------------------------------------
# Physics and simulation (cached)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def simulate(theta1_deg, theta2_deg, m1, m2, L1, L2, g, duration, dt):
    """Integrate double-pendulum equations and return time + coordinates."""
    theta1_0 = np.radians(theta1_deg)
    theta2_0 = np.radians(theta2_deg)
    omega1_0 = 0.0
    omega2_0 = 0.0

    def derivs(state, t):
        theta1, omega1, theta2, omega2 = state
        delta = theta1 - theta2
        denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
        denom2 = (L2 / L1) * denom1

        dtheta1 = omega1
        dtheta2 = omega2

        domega1 = (
            m2 * g * np.sin(theta2) * np.cos(delta)
            - m2 * np.sin(delta) * (L1 * omega1**2 * np.cos(delta) + L2 * omega2**2)
            - (m1 + m2) * g * np.sin(theta1)
        ) / denom1

        domega2 = (
            (m1 + m2)
            * (
                L1 * omega1**2 * np.sin(delta)
                - g * np.sin(theta2)
                + g * np.sin(theta1) * np.cos(delta)
            )
            + m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta)
        ) / denom2

        return [dtheta1, domega1, dtheta2, domega2]

    t = np.arange(0.0, duration, dt)
    sol = odeint(derivs, [theta1_0, omega1_0, theta2_0, omega2_0], t)

    x1 = L1 * np.sin(sol[:, 0])
    y1 = -L1 * np.cos(sol[:, 0])
    x2 = x1 + L2 * np.sin(sol[:, 2])
    y2 = y1 - L2 * np.cos(sol[:, 2])

    return t, x1, y1, x2, y2


# ---------------------------------------------------------------
# Animation builder (blue upper trail, orange lower trail)
# ---------------------------------------------------------------
def build_animation(
    x1, y1, x2, y2,
    show_upper_path=False, show_lower_path=True,
    title="Double Pendulum",
    fps=FIXED_FPS,
    trail_len=None
):
    fig, ax = plt.subplots(figsize=(6, 6))

    max_extent = max(np.max(np.abs(x2)), np.max(np.abs(y2)), 1.2)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)

    rod1, = ax.plot([], [], 'o-', lw=2)
    rod2, = ax.plot([], [], 'o-', lw=2)

    trail_upper = None
    trail_lower = None
    if show_upper_path:
        trail_upper, = ax.plot([], [], '-', lw=1, color='#1f77b4')  # blue
    if show_lower_path:
        trail_lower, = ax.plot([], [], '-', lw=1, color='#ff7f0e')  # orange

    def slice_trail(arr, i):
        if trail_len is None or trail_len <= 0:
            return arr[:i+1]
        start = max(0, i - trail_len + 1)
        return arr[start:i+1]

    def init():
        rod1.set_data([], [])
        rod2.set_data([], [])
        if trail_upper is not None:
            trail_upper.set_data([], [])
        if trail_lower is not None:
            trail_lower.set_data([], [])
        return [rod1, rod2] + ([trail_upper] if trail_upper else []) + ([trail_lower] if trail_lower else [])

    def update(i):
        rod1.set_data([0.0, x1[i]], [0.0, y1[i]])
        rod2.set_data([x1[i], x2[i]], [y1[i], y2[i]])

        if trail_upper is not None:
            trail_upper.set_data(slice_trail(x1, i), slice_trail(y1, i))
        if trail_lower is not None:
            trail_lower.set_data(slice_trail(x2, i), slice_trail(y2, i))

        artists = [rod1, rod2]
        if trail_upper is not None:
            artists.append(trail_upper)
        if trail_lower is not None:
            artists.append(trail_lower)
        return artists

    interval_ms = 1000.0 / fps
    ani = FuncAnimation(
        fig, update, init_func=init,
        frames=len(x1), interval=interval_ms,
        blit=True, repeat=False
    )
    return fig, ani


# ---------------------------------------------------------------
# Angle–time plot (omega1, omega2 as initial angular velocities)
# ---------------------------------------------------------------
def angle_time_plot(theta1_deg, theta2_deg, omega1, omega2, m1, m2, L1, L2, g, duration, points):
    theta1_0 = np.radians(theta1_deg)
    theta2_0 = np.radians(theta2_deg)

    sol = odeint(
        lambda s, t: [
            s[1],
            (
                -m2 * L1 * s[1]**2 * np.sin(s[0] - s[2]) * np.cos(s[0] - s[2])
                + m2 * g * np.sin(s[2]) * np.cos(s[0] - s[2])
                - m2 * L2 * s[3]**2 * np.sin(s[0] - s[2])
                - (m1 + m2) * g * np.sin(s[0])
            ) / ((m1 + m2) * L1 - m2 * L1 * np.cos(s[0] - s[2])**2),
            s[3],
            (
                (m1 + m2)
                * (L1 * s[1]**2 * np.sin(s[0] - s[2])
                   - g * np.sin(s[2])
                   + g * np.sin(s[0]) * np.cos(s[0] - s[2]))
                + m2 * L2 * s[3]**2 * np.sin(s[0] - s[2]) * np.cos(s[0] - s[2])
            ) / ((L2 / L1) * ((m1 + m2) * L1 - m2 * L1 * np.cos(s[0] - s[2])**2))
        ],
        [theta1_0, omega1, theta2_0, omega2],
        np.linspace(0, duration, points)
    )

    time = np.linspace(0, duration, points)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, sol[:, 0], label="Theta1 (upper)")
    ax.plot(time, sol[:, 2], label="Theta2 (lower)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.set_title("Double Pendulum: Angle–Time Behavior")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Interactive Double Pendulum Simulation",
    page_icon="🚀",
    layout="wide"
)
st.title("Interactive Double Pendulum Simulation")

with st.sidebar:
    st.header("Parameters")
    theta1 = st.slider("Theta1 (°)", -180.0, 180.0, 60.0, 1.0)
    theta2 = st.slider("Theta2 (°)", -180.0, 180.0, -30.0, 1.0)
    m1 = st.slider("Mass m1", 0.1, 50.0, 2.0, 0.1)
    m2 = st.slider("Mass m2", 0.1, 50.0, 20.0, 0.1)
    L1 = st.slider("Length L1", 0.1, 5.0, 1.0, 0.1)
    L2 = st.slider("Length L2", 0.1, 5.0, 1.0, 0.1)
    g = st.slider("Gravity g (m/s²)", 1.0, 20.0, 9.81, 0.01)

    duration = st.slider("Duration (s)", 1.0, 15.0, 5.0, 0.5)
    dt = st.slider("Δt (s)", 0.005, 0.1, 0.05, 0.005)
    dt = float(np.round(dt / 0.005) * 0.005)

    show_upper_path = st.checkbox("Show upper path", value=False)
    show_lower_path = st.checkbox("Show lower path", value=True)
    trail_len = st.number_input("Trail length (points, 0 = full)", value=250, min_value=0, step=50)

    outfile_name = st.text_input("Output filename", value="pendulum.gif")
    preview_width = st.slider("Preview width (px)", 600, 1000, 600, 50)

preview_width = int(np.clip(preview_width, 600, 1000))

tabs = st.tabs(["Animation", "Angle–Time Plot"])

with tabs[0]:
    st.subheader("Animation")
    if st.button("Run simulation & render animation"):
        with st.spinner("Simulating and rendering..."):
            t, x1, y1, x2, y2 = simulate(theta1, theta2, m1, m2, L1, L2, g, duration, dt)

            frames_out = int(round(duration * FIXED_FPS))
            idx = np.linspace(0, len(t) - 1, frames_out).round().astype(int)
            idx = np.clip(idx, 0, len(t) - 1)

            fig, ani = build_animation(
                x1[idx], y1[idx], x2[idx], y2[idx],
                show_upper_path=show_upper_path,
                show_lower_path=show_lower_path,
                trail_len=None if trail_len == 0 else int(trail_len)
            )

            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
                gif_path = f.name

            writer = PillowWriter(fps=FIXED_FPS)
            ani.save(gif_path, writer=writer)
            plt.close(fig)

            with open(gif_path, "rb") as f:
                gif_bytes = f.read()
            st.image(gif_bytes, caption="Animation (GIF)", width=preview_width)
            st.download_button("Download GIF", data=gif_bytes, file_name=outfile_name, mime="image/gif")

with tabs[1]:
    st.subheader("Angle–Time Plot")
    omega1 = st.number_input("Initial angular velocity ω1 (rad/s)", value=0.0, step=0.1)
    omega2 = st.number_input("Initial angular velocity ω2 (rad/s)", value=0.0, step=0.1)
    points = st.slider("Number of points", 200, 5000, 1000, 100)

    if st.button("Compute & render plot"):
        with st.spinner("Computing..."):
            png_bytes = angle_time_plot(theta1, theta2, omega1, omega2, m1, m2, L1, L2, g, duration, points)
            st.image(png_bytes, caption="Angle–Time plot", width=preview_width)
            st.download_button("Download PNG", data=png_bytes, file_name="angles.png", mime="image/png")
