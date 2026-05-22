# File: gif_step2_cratered_terrain.py
import pybullet as p
import pybullet_data
import numpy as np
import imageio

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
height_data = np.random.uniform(0, 1, (64, 64)).astype(np.float32).flatten()
terrain = p.createMultiBody(0, p.createCollisionShape(
    shapeType=p.GEOM_HEIGHTFIELD, meshScale=[0.5, 0.5, 2],
    heightfieldData=height_data, numHeightfieldRows=64, numHeightfieldColumns=64))
# p.loadTexture("checker_grid.jpg") This texture file needs to exist for the script to run. This texture comes bundled with PyBullet data
p.changeVisualShape(
    terrain, -1, textureUniqueId=p.loadTexture("checker_grid.jpg"))
rover = p.loadURDF("r2d2.urdf", [1, 1, 1])p.loadTexture
frames = []

for _ in range(100):
    p.stepSimulation()
    pos, _ = p.getBasePositionAndOrientation(rover)
    p.resetBasePositionAndOrientation(
        rover, [pos[0]+0.03, pos[1], pos[2]], [0, 0, 0, 1])
    _, _, img, *_ = p.getCameraImage(320, 240,
                                     p.computeViewMatrixFromYawPitchRoll(
                                         [2, 2, 1], 4, 60, -30, 0, 2),
                                     p.computeProjectionMatrixFOV(
                                         60, 320/240, 0.1, 100),
                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)
    frames.append(np.reshape(img, (240, 320, 4))[:, :, :3].astype(np.uint8))
p.disconnect()

imageio.mimsave("gif_step2_cratered_terrain.gif", frames, fps=15)
