import pybullet as p
import pybullet_data
import numpy as np
import cv2
import os

# Connect to PyBullet in DIRECT mode
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# Load cratered terrain using a synthetic heightfield
terrain_shape = p.createCollisionShape(
    shapeType=p.GEOM_HEIGHTFIELD,
    meshScale=[0.5, 0.5, 2],
    heightfieldTextureScaling=128,
    heightfieldData=np.random.uniform(0, 1, 256).astype(np.float32),
    numHeightfieldRows=16,
    numHeightfieldColumns=16
)
terrain = p.createMultiBody(0, terrain_shape)

# Apply texture
p.changeVisualShape(
    # p.loadTexture("checker_grid.jpg") This texture file needs to exist for the script to run. This texture comes bundled with PyBullet data
    terrain, -1, textureUniqueId=p.loadTexture("checker_grid.jpg"))

# Load R2D2 as a placeholder rover
r2d2_id = p.loadURDF("r2d2.urdf", [0, 0, 1])

# Set up camera
camera_distance = 5
camera_yaw = 45
camera_pitch = -30
camera_target_position = [0, 0, 0.5]

# Simulation frames
frames = []
print("[INFO] Simulating rover over cratered terrain...")

for _ in range(120):
    p.stepSimulation()
    pos, _ = p.getBasePositionAndOrientation(r2d2_id)
    p.resetBasePositionAndOrientation(
        r2d2_id, [pos[0] + 0.05, pos[1], pos[2]], [0, 0, 0, 1])

    # Render camera frame
    width, height, rgb_img, *_ = p.getCameraImage(
        width=320,
        height=240,
        viewMatrix=p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target_position,
            distance=camera_distance,
            yaw=camera_yaw,
            pitch=camera_pitch,
            roll=0,
            upAxisIndex=2),
        projectionMatrix=p.computeProjectionMatrixFOV(
            fov=60, aspect=320 / 240, nearVal=0.1, farVal=100.0),
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb = np.reshape(rgb_img, (height, width, 4))[:, :, :3]
    rgb_uint8 = rgb.astype(np.uint8)
    frame_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    frames.append(frame_bgr)

p.disconnect()

# Save the video
os.makedirs("outputs", exist_ok=True)
video_path = "outputs/rover_cratered_demo.mp4"
print(f"[INFO] Saving video to {video_path}...")

out = cv2.VideoWriter(
    video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (320, 240))
for frame in frames:
    out.write(frame)
out.release()

print("[SUCCESS] Cratered terrain demo video saved as rover_cratered_demo.mp4")
