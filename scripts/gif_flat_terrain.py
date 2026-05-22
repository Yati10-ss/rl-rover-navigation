import pybullet as p
import pybullet_data
import numpy as np
import imageio
import os

# Setup
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")
rover = p.loadURDF("r2d2.urdf", [0, 0, 0.5])

frames = []

# Simulate forward movement
for step in range(80):
    p.stepSimulation()
    pos, _ = p.getBasePositionAndOrientation(rover)
    p.resetBasePositionAndOrientation(
        rover, [pos[0] + 0.05, pos[1], pos[2]], [0, 0, 0, 1])
    width, height, rgb, *_ = p.getCameraImage(
        width=320,
        height=240,
        viewMatrix=p.computeViewMatrixFromYawPitchRoll(
            [1, 1, 1], 3, 45, -30, 0, 2),
        projectionMatrix=p.computeProjectionMatrixFOV(
            60, 320 / 240, 0.1, 100.0),
        renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_np = np.reshape(rgb, (240, 320, 4))[:, :, :3].astype(np.uint8)
    frames.append(rgb_np)

p.disconnect()

# Save GIF
os.makedirs("outputs", exist_ok=True)
gif_path = "outputs/gif_step1_flat_terrain.gif"
imageio.mimsave(gif_path, frames, fps=15)
print(f"[✅] GIF saved: {gif_path}")
