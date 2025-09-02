from real_robot_env.robot.hardware_depthai import DepthAI
from real_robot_env.robot.hardware_cameras import AsynchronousCamera
import time
from pathlib import Path

cam = AsynchronousCamera[DepthAI](
    camera_class=DepthAI,
    capture_interval=0.05,
    device_id="1844301051D9B50F00",
    name="depthai_camera",
)

print("Camera initialized:", cam)

cam.connect()

timestamps = []
timestamps.append(time.time())
print("Camera connected:", cam)
print("Waiting for camera to start...")
time.sleep(1)
timestamps.append(time.time())
time.sleep(4)
cam.start_recording()
print("Recording started...")

for i in range(100):
    time.sleep(0.1)
    timestamps.append(time.time())

cam.stop_recording()
cam.store_recording(
    directory=Path("/home/kkuryshev/audio-pipeline/real_robot/test"),
    timestamps=timestamps,
)
