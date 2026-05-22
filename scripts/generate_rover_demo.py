import pybullet as p
import pybullet_data
import numpy as np
import cv2
import os

# Connect to PyBullet in headless mode
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# Load flat plane and R2D2 robot
p.loadURDF("plane.urdf")
r2d2_id = p.loadURDF("r2d2.urdf", [0, 0, 0.1])

# Camera settings for rendering
camera_distance = 3
camera_yaw = 50
camera_pitch = -35
camera_target_position = [0, 0, 0]

# Prepare to collect frames
frames = []
print("[INFO] Simulating rover movement...")

for _ in range(120):
    p.stepSimulation()

    # Move R2D2 slightly forward in X-direction
    pos, _ = p.getBasePositionAndOrientation(r2d2_id)
    p.resetBasePositionAndOrientation(
        r2d2_id, [pos[0] + 0.01, pos[1], pos[2]], [0, 0, 0, 1])

    # Capture a frame from the camera
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

    # Convert to uint8 format and save frame
    rgb = np.reshape(rgb_img, (height, width, 4))[:, :, :3]
    rgb_uint8 = rgb.astype(np.uint8)
    frame_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    frames.append(frame_bgr)

p.disconnect()

# Save video to file
# Use raw string if saving to custom path
os.makedirs("outputs", exist_ok=True)
video_path = "outputs/rover_demo_final.mp4"
print(f"[INFO] Saving video to {video_path}...")

out = cv2.VideoWriter(
    video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (320, 240))
for frame in frames:
    out.write(frame)
out.release()

print("[SUCCESS] Rover demo video saved as rover_demo_final.mp4")
