import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio

# Simulate trajectory data
positions = np.cumsum(np.random.randn(100, 2) * 0.2, axis=0)

frames = []
for i in range(10, 100, 5):
    fig, ax = plt.subplots(figsize=(5, 4))
    canvas = FigureCanvas(fig)  # Explicit Agg backend
    ax.plot(positions[:i, 0], positions[:i, 1], marker='o', markersize=3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(f"Trajectory till step {i}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    frames.append(img)
    plt.close(fig)

# Save the animated GIF
imageio.mimsave("gif_step6_trajectory.gif", frames, fps=3)
print("[✅] Saved: gif_step6_trajectory.gif")
