
#!/usr/bin/env python3
"""
Streamlit GUI for Interactive Simulation and Visualization of Chaotic Nonlinear Behavior of a Double Pendulum Using Python:
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


# ------------------------------
# Physics and simulation (cached)
# ------------------------------
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

    # Number of points is governed by duration and dt
    t = np.arange(0.0, duration, dt)
    initial_state = [theta1_0, omega1_0, theta2_0, omega2_0]
    sol = odeint(derivs, initial_state, t)

    x1 = L1 * np.sin(sol[:, 0])
    y1 = -L1 * np.cos(sol[:, 0])
    x2 = x1 + L2 * np.sin(sol[:, 2])
    y2 = y1 - L2 * np.cos(sol[:, 2])
    return t, x1, y1, x2, y2


# ------------------------------
# Utility: optional frame downsampling
# ------------------------------
def downsample_series(t, x1, y1, x2, y2, stride):
    """Return series sliced by the given positive integer stride."""
    stride = max(1, int(stride))
    return t[::stride], x1[::stride], y1[::stride], x2[::stride], y2[::stride]


# ------------------------------
# Animation builder
# ------------------------------
def build_animation(
    t, x1, y1, x2, y2,
    show_upper_path=False, show_lower_path=True,
    title="Double Pendulum",
    preview_fps=30,
    repeat_preview=True
):
    fig, ax = plt.subplots(figsize=(6, 6))

    max_extent = max(np.max(np.abs(x2)), np.max(np.abs(y2)), 1.0)
    max_extent = max(max_extent, 1.2)
    x_min, x_max, y_min, y_max = (-max_extent, max_extent, -max_extent, max_extent)

    trail1_x, trail1_y = [], []
    trail2_x, trail2_y = [], []

    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        if show_upper_path:
            trail1_x.append(x1[frame]); trail1_y.append(y1[frame])
            ax.plot(trail1_x, trail1_y,
                    linestyle='-', linewidth=1, color='#1f77b4', label='Upper path')

        if show_lower_path:
            trail2_x.append(x2[frame]); trail2_y.append(y2[frame])
            ax.plot(trail2_x, trail2_y,
                    linestyle='-', linewidth=1, color='#ff7f0e', label='Lower path')

        ax.plot([0, x1[frame]], [0, y1[frame]], 'o-', lw=2, color='#1f77b4')
        ax.plot([x1[frame], x2[frame]], [y1[frame], y2[frame]], 'o-', lw=2, color='#ff7f0e')
        ax.set_title(title)
        if show_upper_path or show_lower_path:
            ax.legend(loc='upper left', frameon=False)

    # Preview interval derived from desired preview fps (ms per frame)
    interval_ms = 1000.0 / max(1, int(round(preview_fps)))
    ani = FuncAnimation(fig, update, frames=len(t), interval=interval_ms, repeat=repeat_preview)
    return fig, ani


# ------------------------------
# Angle–time plot
# ------------------------------
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
            - m2 * L1 * z1_**2 * np.sin(delta) * np.cos(delta)
            + m2 * g * np.sin(theta2) * np.cos(delta)
            - m2 * L2 * z2_**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(theta1)
        ) / denominator1

        dz2_dt = (
            m2 * L2 * z2_**2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
            + (m1 + m2) * L1 * z1_**2 * np.sin(delta)
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


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(
    page_title="Interactive Simulation and Visualization of Chaotic Nonlinear Behavior of a Double Pendulum Using Python (Streamlit GUI)",
    page_icon="🪀",
    layout="wide"
)

st.title("🪀 Interactive Simulation and Visualization of Chaotic Nonlinear Behavior of a Double Pendulum Using Python")

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

    # Playback speed control: Slower / Real / Faster
    speed_label = st.select_slider(
        "Playback speed",
        options=["Slower", "Real", "Faster"],
        value="Real",
        help="Controls playback relative to real time; simulation density still comes from Δt."
    )
    speed_factor = {"Slower": 0.5, "Real": 1.0, "Faster": 2.0}[speed_label]

    st.divider()
    show_upper_path = st.checkbox("Show upper path", value=False)
    show_lower_path = st.checkbox("Show lower path", value=True)

    st.divider()
    writer_label = st.radio("Output format", ["GIF (Pillow)", "MP4 (FFmpeg)"], index=0)
    outfile_name = st.text_input(
        "Output filename",
        value="pendulum.gif" if writer_label.startswith("GIF") else "pendulum.mp4"
    )

    st.divider()
    preview_width = st.slider("Preview width (px)", 600, 1000, 600, 50)

# Clamp preview width to be extra‑safe
preview_width = int(np.clip(preview_width, 600, 1000))

tabs = st.tabs(["🎞️ Animation", "📈 Angle–Time Plot"])

with tabs[0]:
    st.subheader("Animation")
    run_anim = st.button("Run simulation & render animation")

    if run_anim:
        with st.spinner("Simulating and rendering..."):
            # --- Simulate
            t, x1, y1, x2, y2 = simulate(theta1, theta2, m1, m2, L1, L2, g, duration, dt)

            # --- Derive timing from simulation so "Real" lasts exactly `duration`
            # Base fps from simulation samples (N/duration)
            base_fps = max(1, int(round(len(t) / duration)))

            # Apply speed factor (Slower/Real/Faster)
            target_fps = base_fps * speed_factor

            # Optional: cap preview fps and downsample frames to keep things smooth
            max_preview_fps = 60  # UI smoothness cap
            stride = max(1, int(np.ceil(target_fps / max_preview_fps)))

            # Downsample series if needed (affects both preview and export)
            t_s, x1_s, y1_s, x2_s, y2_s = downsample_series(t, x1, y1, x2, y2, stride)

            # Recompute base fps after downsampling to preserve correct timing
            base_fps_s = max(1, int(round(len(t_s) / duration)))
            preview_fps = min(int(round(base_fps_s * speed_factor)), max_preview_fps)
            writer_fps = max(1, int(round(base_fps_s * speed_factor)))

            # --- Build preview (repeat=True so it loops in the UI)
            fig, ani = build_animation(
                t_s, x1_s, y1_s, x2_s, y2_s,
                show_upper_path, show_lower_path,
                title="Double Pendulum",
                preview_fps=preview_fps,
                repeat_preview=True
            )

            # --- Save/export
            ext = os.path.splitext(outfile_name)[1].lower()
            use_gif = writer_label.startswith("GIF") or ext == ".gif"

            if use_gif:
                with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
                    gif_path = f.name
                writer = PillowWriter(fps=writer_fps)
                ani.save(gif_path, writer=writer)
                plt.close(fig)
                st.image(gif_path, caption="Animation (GIF)", width=preview_width)
                with open(gif_path, "rb") as f:
                    st.download_button(
                        "Download GIF",
                        data=f.read(),
                        file_name=outfile_name or "pendulum.gif",
                        mime="image/gif"
                    )
            else:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    mp4_path = f.name
                writer = FFMpegWriter(fps=writer_fps)
                ani.save(mp4_path, writer=writer)
                plt.close(fig)
                st.video(mp4_path)
                with open(mp4_path, "rb") as f:
                    st.download_button(
                        "Download MP4",
                        data=f.read(),
                        file_name=outfile_name or "pendulum.mp4",
                        mime="video/mp4"
                    )

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
