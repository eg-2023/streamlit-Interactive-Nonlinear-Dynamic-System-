
#!/usr/bin/env python3
"""
Streamlit GUI for double pendulum:
- Animation (GIF via Pillow; MP4 via FFmpeg)
- Angle–time plot (Appendix A style)
Run: streamlit run app.py
"""

import os
import io
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for Streamlit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.integrate import odeint
import streamlit as st


# ---------------------------------------
# Physics and simulation (cached)
# ---------------------------------------
@st.cache_data(show_spinner=False)
def simulate(theta1_deg, theta2_deg, m1, m2, L1, L2, g, duration, dt):
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
            - m2 * np.sin(delta) * (L1 * omega1 ** 2 * np.cos(delta) + L2 * omega2 ** 2)
            - (m1 + m2) * g * np.sin(theta1)
        ) / denom1

        domega2 = (
            (m1 + m2) * (L1 * omega1 ** 2 * np.sin(delta)
                         - g * np.sin(theta2)
                         + g * np.sin(theta1) * np.cos(delta))
            + m2 * L2 * omega2 ** 2 * np.sin(delta) * np.cos(delta)
        ) / denom2

        return [dtheta1, domega1, dtheta2, domega2]

    t = np.arange(0.0, duration, dt)
    initial_state = [theta1_0, omega1_0, theta2_0, omega2_0]
    sol = odeint(derivs, initial_state, t)

    x1 = L1 * np.sin(sol[:, 0])
    y1 = -L1 * np.cos(sol[:, 0])
    x2 = x1 + L2 * np.sin(sol[:, 2])
    y2 = y1 - L2 * np.cos(sol[:, 2])

    return t, x1, y1, x2, y2


# ---------------------------------------
# Animation builder
# ---------------------------------------
def build_animation(t, x1, y1, x2, y2,
                    show_upper_path=False, show_lower_path=True,
                    title="Double Pendulum"):
    fig, ax = plt.subplots(figsize=(6, 6))

    max_extent = max(np.max(np.abs(x2)), np.max(np.abs(y2)), 1.0)
    max_extent = max(max_extent, 1.2)
    limits = (-max_extent, max_extent, -max_extent, max_extent)
    x_min, x_max, y_min, y_max = limits

    trail1_x, trail1_y = [], []
    trail2_x, trail2_y = [], []

    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        if show_upper_path:
            trail1_x.append(x1[frame])
            trail1_y.append(y1[frame])
            ax.plot(trail1_x, trail1_y, 'b--', linewidth=1, label='Upper path')
        if show_lower_path:
            trail2_x.append(x2[frame])
            trail2_y.append(y2[frame])
            ax.plot(trail2_x, trail2_y, 'r-', linewidth=1, label='Lower path')

        ax.plot([0, x1[frame]], [0, y1[frame]], 'o-', lw=2, color='#1f77b4')
        ax.plot([x1[frame], x2[frame]], [y1[frame], y2[frame]], 'o-', lw=2, color='#ff7f0e')
        ax.set_title(title)

        if show_upper_path or show_lower_path:
            ax.legend(loc='upper left', frameon=False)

    ani = FuncAnimation(fig, update, frames=len(t), interval=(t[1] - t[0]) * 1000, repeat=False)
    return fig, ani


# ---------------------------------------
# Angle–time plot
# ---------------------------------------
def angle_time_plot(theta1_deg, theta2_deg, z1, z2, m1, m2, L1, L2, g, duration, points):
    theta1_0 = np.radians(theta1_deg)
    theta2_0 = np.radians(theta2_deg)
    state0 = [theta1_0, z1, theta2_0, z2]

    def equations(state, t):
        theta1, z1_, theta2, z2_ = state
        delta = theta2 - theta1
        denominator1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
        denominator2 = (L2 / L1) * denominator1

        dtheta1_dt = z1_
        dtheta2_dt = z2_

        dz1_dt = (
            - m2 * L1 * z1_ ** 2 * np.sin(delta) * np.cos(delta)
            + m2 * g * np.sin(theta2) * np.cos(delta)
            - m2 * L2 * z2_ ** 2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(theta1)
        ) / denominator1

        dz2_dt = (
            m2 * L2 * z2_ ** 2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
            + (m1 + m2) * L1 * z1_ ** 2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(theta2)
        ) / denominator2

        return [dtheta1_dt, dz1_dt, dtheta2_dt, dz2_dt]

    time = np.linspace(0.0, duration, points)
    sol = odeint(equations, state0, time)

    theta1 = (sol[:, 0] + np.pi) % (2 * np.pi) - np.pi
    theta2 = (sol[:, 2] + np.pi) % (2 * np.pi) - np.pi

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, theta1, label="Theta1 (upper)")
    ax.plot(time, theta2, label="Theta2 (lower)")
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


# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.set_page_config(page_title="Double Pendulum (Streamlit GUI)", page_icon="🪀", layout="wide")

st.title("🪀 Double Pendulum — Streamlit GUI")

with st.sidebar:
    st.header("Parameters")

    theta1 = st.slider("Theta1 (°)", -180.0, 180.0, 60.0, 1.0)
    theta2 = st.slider("Theta2 (°)", -180.0, 180.0, -30.0, 1.0)

    m1 = st.slider("Mass m1", 0.1, 50.0, 2.0, 0.1)
    m2 = st.slider("Mass m2", 0.1, 50.0, 20.0, 0.1)

    L1 = st.slider("Length L1", 0.1, 5.0, 1.0, 0.1)
    L2 = st.slider("Length L2", 0.1, 5.0, 1.0, 0.1)

    g = st.slider("Gravity g (m/s²)", 1.0, 20.0, 9.81, 0.01)

    st.divider()
    duration = st.slider("Duration (s)", 1.0, 30.0, 5.0, 0.5)
    dt = st.slider("Δt (s)", 0.005, 0.1, 0.05, 0.005)
    fps = st.slider("FPS", 5, 60, 15, 1)

    st.divider()
    show_upper_path = st.checkbox("Show upper path", value=False)
    show_lower_path = st.checkbox("Show lower path", value=True)

    st.divider()
    writer_label = st.radio("Output format", ["GIF (Pillow)", "MP4 (FFmpeg)"], index=0)
    outfile_name = st.text_input("Output filename", value="pendulum.gif" if writer_label.startswith("GIF") else "pendulum.mp4")

    st.divider()
    preview_width = st.slider("Preview width (px)", 300, 1000, 600, 50)

tabs = st.tabs(["🎞️ Animation", "📈 Angle–Time Plot"])

with tabs[0]:
    st.subheader("Animation")
    run_anim = st.button("Run simulation & render animation")

    if run_anim:
        with st.spinner("Simulating and rendering..."):
            t, x1, y1, x2, y2 = simulate(theta1, theta2, m1, m2, L1, L2, g, duration, dt)
            fig, ani = build_animation(t, x1, y1, x2, y2, show_upper_path, show_lower_path)

            ext = os.path.splitext(outfile_name)[1].lower()
            use_gif = writer_label.startswith("GIF") or ext == ".gif"

            if use_gif:
                with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
                    gif_path = f.name
                writer = PillowWriter(fps=fps)
                ani.save(gif_path, writer=writer)
                plt.close(fig)

                st.image(gif_path, caption="Animation (GIF)", width=preview_width)
                with open(gif_path, "rb") as f:
                    st.download_button("Download GIF", data=f.read(), file_name=outfile_name or "pendulum.gif", mime="image/gif")
            else:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    mp4_path = f.name
                writer = FFMpegWriter(fps=fps)
                ani.save(mp4_path, writer=writer)
                plt.close(fig)

                st.video(mp4_path)
                with open(mp4_path, "rb") as f:
                    st.download_button("Download MP4", data=f.read(), file_name=outfile_name or "pendulum.mp4", mime="video/mp4")

with tabs[1]:
    st.subheader("Angle–Time Plot")
    z1 = st.number_input("Initial angular velocity z1 (rad/s)", value=0.0, step=0.1)
    z2 = st.number_input("Initial angular velocity z2 (rad/s)", value=0.0, step=0.1)
    points = st.slider("Number of points", 200, 5000, 1000, 100)

    run_plot = st.button("Compute & render plot")

    if run_plot:
        with st.spinner("Computing..."):
            png_bytes = angle_time_plot(theta1, theta2, z1, z2, m1, m2, L1, L2, g, duration, points)
        st.image(png_bytes, caption="Angle–Time plot", width=preview_width)
        st.download_button("Download PNG", data=png_bytes, file_name="angles.png", mime="image/png")

st.caption("Tip: GIF works without FFmpeg. For MP4, install FFmpeg and choose MP4 in the sidebar.")

