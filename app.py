
#!/usr/bin/env python3
"""
Streamlit GUI for Interactive Simulation and Visualization of Chaotic Nonlinear Behavior of a Double Pendulum Using Python:
- Animation (GIF via Pillow; MP4 via FFmpeg), fixed at 50 FPS
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
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.integrate import odeint
import streamlit as st

FIXED_FPS = 50  # match typical browser GIF min-frame-delay (~20 ms) → ~50 FPS

# ---------------------------------------------------------------
# Physics and simulation (cached)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def simulate(theta1_deg, theta2_deg, m1, m2, L1, L2, g, duration, dt):
    """Integrate double-pendulum equations and return time + coordinates sampled at dt."""
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

    # Samples governed by duration and dt (endpoint excluded)
    t = np.arange(0.0, duration, dt)
    initial_state = [theta1_0, omega1_0, theta2_0, omega2_0]
    sol = odeint(derivs, initial_state, t)

    x1 = L1 * np.sin(sol[:, 0])
    y1 = -L1 * np.cos(sol[:, 0])
    x2 = x1 + L2 * np.sin(sol[:, 2])
    y2 = y1 - L2 * np.cos(sol[:, 2])

    return t, x1, y1, x2, y2

# ---------------------------------------------------------------
# Deterministic resampling to exact frame count
# ---------------------------------------------------------------
def resample_indices(n_in: int, frames_out: int) -> np.ndarray:
    """Return integer indices (len = frames_out) evenly spanning [0, n_in-1]."""
    frames_out = max(1, int(frames_out))
    if n_in <= 1:
        return np.array([0] * frames_out, dtype=int)
    idx = np.linspace(0, n_in - 1, frames_out)
    return np.clip(idx.round().astype(int), 0, n_in - 1)

# ---------------------------------------------------------------
# Animation builder (blitting + exact duration via fixed 50 FPS)
# ---------------------------------------------------------------
def build_animation(
    x1, y1, x2, y2,
    show_upper_path=False, show_lower_path=True,
    title="Double Pendulum",
    fps=FIXED_FPS,
    trail_len=None,       # None → full trail; else show last N points
    repeat_preview=True
):
    """Create a blitted animation. Inputs are already resampled to desired frame count."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Limits
    max_extent = max(np.max(np.abs(x2)), np.max(np.abs(y2)), 1.2)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)

    # Artists: rods and trails
    rod1, = ax.plot([], [], 'o-', lw=2, color='#1f77b4', zorder=3)  # origin -> (x1,y1)
    rod2, = ax.plot([], [], 'o-', lw=2, color='#ff7f0e', zorder=3)  # (x1,y1) -> (x2,y2)

    trail_upper = None
    trail_lower = None
    if show_upper_path:
        trail_upper, = ax.plot([], [], '-', lw=1, color='#1f77b4', zorder=1, label='Upper path')
    if show_lower_path:
        trail_lower, = ax.plot([], [], '-', lw=1, color='#ff7f0e', zorder=1, label='Lower path')
    if show_upper_path or show_lower_path:
        ax.legend(loc='upper left', frameon=False)

    # Precompute trail slicing helper
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
        artists = [rod1, rod2]
        if trail_upper is not None: artists.append(trail_upper)
        if trail_lower is not None: artists.append(trail_lower)
        return artists

    def update(i):
        # Rods
        rod1.set_data([0.0, x1[i]], [0.0, y1[i]])
        rod2.set_data([x1[i], x2[i]], [y1[i], y2[i]])

        # Trails
        if trail_upper is not None:
            trail_upper.set_data(slice_trail(x1, i), slice_trail(y1, i))
        if trail_lower is not None:
            trail_lower.set_data(slice_trail(x2, i), slice_trail(y2, i))

        artists = [rod1, rod2]
        if trail_upper is not None: artists.append(trail_upper)
        if trail_lower is not None: artists.append(trail_lower)
        return artists

    interval_ms = 1000.0 / fps
    ani = FuncAnimation(
        fig, update, init_func=init,
        frames=len(x1), interval=interval_ms,
        blit=True, repeat=repeat_preview
    )
    return fig, ani

# ---------------------------------------------------------------
# Angle–time plot
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Interactive Simulation and Visualization of Chaotic Nonlinear Behavior of a Double Pendulum Using Python (Streamlit GUI)",
    page_icon="🚀",
    layout="wide"
)
st.title("🚀 Interactive Simulation and Visualization of Chaotic Nonlinear Behavior of a Double Pendulum Using Python")

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

    # dt slider: format + quantize to avoid duplicate-looking values
    dt = st.slider("Δt (s)", 0.005, 0.1, 0.05, 0.005, format="%.3f")
    dt = float(np.round(dt / 0.005) * 0.005)

    st.divider()
    show_upper_path = st.checkbox("Show upper path", value=False)
    show_lower_path = st.checkbox("Show lower path", value=True)

    # Optional: limit the trail length to keep blitted updates light (0 = full trail)
    trail_len = st.number_input("Trail length (points, 0 = full)", value=250, min_value=0, step=50)

    st.divider()
    writer_label = st.radio("Output format", ["GIF (Pillow)", "MP4 (FFmpeg)"], index=0)
    # GIF loop control (most browsers honor this; 0 = infinite, 1 = play once)
    loop_gif = st.checkbox("Loop GIF in preview", value=True)

    outfile_name = st.text_input("Output filename", value="pendulum.gif" if writer_label.startswith("GIF") else "pendulum.mp4")

    st.divider()
    preview_width = st.slider("Preview width (px)", 600, 1000, 600, 50)

# Clamp preview width (safety)
preview_width = int(np.clip(preview_width, 600, 1000))

tabs = st.tabs(["🎞️ Animation", "📈 Angle–Time Plot"])

with tabs[0]:
    st.subheader("Animation (fixed 50 FPS)")
    run_anim = st.button("Run simulation & render animation")
    if run_anim:
        with st.spinner("Simulating and rendering..."):
            # 1) Simulate physics with the chosen dt
            t, x1, y1, x2, y2 = simulate(theta1, theta2, m1, m2, L1, L2, g, duration, dt)

            # 2) Resample deterministically to exactly frames_out = round(Duration × 50)
            frames_out = int(round(duration * FIXED_FPS))
            idx = resample_indices(len(t), frames_out)
            x1_s, y1_s, x2_s, y2_s = x1[idx], y1[idx], x2[idx], y2[idx]

            # 3) Build blitted animation (fast & predictable)
            fig, ani = build_animation(
                x1_s, y1_s, x2_s, y2_s,
                show_upper_path=show_upper_path,
                show_lower_path=show_lower_path,
                title="Double Pendulum",
                fps=FIXED_FPS,
                trail_len=None if trail_len == 0 else int(trail_len),
                repeat_preview=loop_gif if writer_label.startswith("GIF") else True
            )

            # 4) Save & preview (use bytes to avoid browser caching)
            use_gif = writer_label.startswith("GIF") or os.path.splitext(outfile_name)[1].lower() == ".gif"

            if use_gif:
                with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
                    gif_path = f.name
                loop_meta = 0 if loop_gif else 1  # 0 = loop, 1 = play once
                writer = PillowWriter(fps=FIXED_FPS, metadata={'loop': loop_meta})
                ani.save(gif_path, writer=writer)
                plt.close(fig)

                with open(gif_path, "rb") as f:
                    gif_bytes = f.read()
                st.image(gif_bytes, caption=f"Animation (GIF, {FIXED_FPS} FPS)", width=preview_width)

                with open(gif_path, "rb") as f:
                    st.download_button("Download GIF", data=f.read(),
                                       file_name=outfile_name or "pendulum.gif", mime="image/gif")

            else:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    mp4_path = f.name
                writer = FFMpegWriter(fps=FIXED_FPS)
                ani.save(mp4_path, writer=writer)
                plt.close(fig)

                with open(mp4_path, "rb") as f:
                    mp4_bytes = f.read()
                st.video(mp4_bytes)

                with open(mp4_path, "rb") as f:
                    st.download_button("Download MP4", data=f.read(),
                                       file_name=outfile_name or "pendulum.mp4", mime="video/mp4")

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

# Informative tip about GIF frame-delay limits and performance
st.caption(
    "Note: GIF playback in browsers is typically capped by a ~20 ms minimum frame delay (≈ 50 FPS). "
    "MP4 respects the encoded FPS. Uses blitting for fast rendering."
)
