# File: gif_step3_observations.py
import pybullet as p, pybullet_data, numpy as np, imageio

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")
rover = p.loadURDF("r2d2.urdf", [0, 0, 0.5])
frames = []

for _ in range(80):
    p.stepSimulation()
    pos, _ = p.getBasePositionAndOrientation(rover)
    p.resetBasePositionAndOrientation(rover, [pos[0]+0.04, pos[1], pos[2]], [0,0,0,1])
    _, _, img, *_ = p.getCameraImage(320,240,
        p.computeViewMatrixFromYawPitchRoll([pos[0], pos[1], 5], 5, 0, -90, 0, 2),
        p.computeProjectionMatrixFOV(60, 320/240, 0.1, 100),
        renderer=p.ER_BULLET_HARDWARE_OPENGL)
    frames.append(np.reshape(img, (240,320,4))[:,:,:3].astype(np.uint8))
p.disconnect()

imageio.mimsave("gif_step3_observation_space.gif", frames, fps=15)
