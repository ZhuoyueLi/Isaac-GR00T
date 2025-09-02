# Installation

1. Install and compile [Polymetis](https://github.com/intuitive-robots/irl_polymetis) (on workstation part)
    - Add `mkl==2024.0.0` in `polymetis/environment.yml`
2. ``` pip install -r requirements.txt ```

Additionally install the dependencies for whatever cameras/recording devices you need as follows.

## Dependencies for Cameras

First install moviepy for camera capture:
```
pip install --trusted-host pypi.python.org moviepy==1.0.3
pip install imageio-ffmpeg
```
In certain versions of moviepy, editing the [following line](https://github.com/intuitive-robots/real_robot/blob/b0c9c78455cd638cb11641e9da8df4762415e120/real_robot_env/robot/hardware_cameras.py#L28) to
`from moviepy import VideoFileClip` helps.

- Install depthai for use of DepthAI (OAK-D) cameras:
```
pip install depthai==2.27.0.0
```
- Install [pykinect_azure](https://github.com/ibaiGorordo/pyKinectAzure) for use of Azure cameras:
```
pip install pykinect_azure==0.0.3
```
- Install [pyrealsense2](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python) for use of RealSense cameras:
```
pip install pyrealsense2==2.55.1.6486
```
- Install [gopro-py-api](https://github.com/KonradIT/gopro-py-api) for use of GoPro Hero 9 Black:
```
pip install goprocam==4.2.0
```

## Dependencies for other devices

- Install [PyAudio](https://pypi.org/project/PyAudio/) for audio recording:
```
pip install PyAudio==0.2.14
```
- Install [digit-interface](https://github.com/facebookresearch/digit-interface) for use of the DIGIT tactile sensors:
```
pip install digit-interface==0.2.1
```

# Usage

<!-- ## Robot PC

- Open Desk for P3 and P4 (can be found in the bookmarks)
- For both of the robots:
    - Unlock the joints
    - Activate FCI
    
- Run the following commands on different terminals:
    - Run P3 robot server
      ```
      mamba activate polymetis
      launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=172.16.3.2 port=50051
      ```

    - Run P3 gripper server
      ```
      mamba activate polymetis
      launch_gripper.py gripper_client=franka_hand gripper.executable_cfg.robot_ip=172.16.3.2 port=50052
      ```

    - Run P4 robot server
      ```
      mamba activate polymetis
      launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=172.16.4.2 port=50053
      ```

    - Run P4 gripper server
      ```
      mamba activate polymetis
      launch_gripper.py gripper_client=franka_hand gripper.executable_cfg.robot_ip=172.16.4.2 port=50054
      ``` -->

## Control robots directly from workstation PC

### Unlock robots, activate FCI and run robot servers
```
bash polymetis_scripts/start_servers.sh
```

### Lock the robots
```
bash polymetis_scripts/lock_robots.sh
```

### Activate environment
``` 
mamba activate robo
```

### Move robots
```
python move_robots.py <robots>
```
e.g
```
python move_robots.py p3 p4
```

## Gymnasium environment

If you need a Gym environment as interface, you can use the class `RealRobotEnv` in `real_robot_env/real_robot_env.py` as reference. The method `__get_obs()` can be used to modify the observation.

Alternatively, you can run a Gym environment through a ZMQ server with the files in `real_robot_env/env_server/`. This allows you to run a ZMQ server with the Real Robot dependencies (e.g. Polymetis) and in a different Python environment you can run the `EnvClient`. For instance, when setting up policies for evaluation on a robot, this can be helpful to deal with dependency conflicts.

Use the Hydra Configs in `configs/env_server/` as reference to define your Gym environment. Then launch it with `python ./launch_env_server.py +env_server=panda_102_octo` where `panda_102_octo` is your config file. To access the server in your other repository, copy the file `real_robot_env/env_server/env_client.py` and optionally adapt it to your needs.

## Data collection using teleoperation

Data collection using teleoperation is implemented in the script `collect_data.py`. The core logic and user interface is in the class `DataCollectionManager`. This script uses [Hydra configs](https://hydra.cc/docs/intro/) and its automatic instantiation of objects to provide a general way to define differet lab environments. You can launch the data collection e.g. with:

```
python ./collect_data.py +collect_data=my_environment
```

`my_environment` is a main config file that specifies the environment that you are using. For more examples that you can use as reference to create config files, see the folder `configs/collect_data/`. 

### GELLO

If you want to use Gello for controlling the robot, check out this [repository](https://github.com/intuitive-robots/gello_software_irl). For instance, in the kitchen environment (Panda 102), you can use the following command after you started the Gello server:

```
python ./collect_data.py +collect_data=panda_102_gello
```

### Config Structure

The root folder `configs/` contains the folder `collect_data/` with main configuration files of the data collection script. Each config file in there specifies a lab environment, among other things this includes the teleoperation pairs and devices used for recording. It is recommended to import hardware files from the central folder `configs/hardware`, which allows to reuse hardware specific settings.

### Devices
  
**Manually creating continuous and discrete devices**:

```python
from real_robot_env.robot.hardware_azure import Azure
from real_robot_env.robot.hardware_depthai import DepthAI
from real_robot_env.robot.hardware_realsense import RealSense
from real_robot_env.robot.hardware_digit import TactileDigit
from real_robot_env.robot.hardware_gopro import GoPro
from real_robot_env.robot.hardware_audio import AudioInterface

# Discrete devices: Azure, DethAI, and RealSense cameras, Digit tactile sensors
azure_cameras = Azure.get_devices(amount)
dai_cameras = DepthAI.get_devices()
rs_cameras = RealSense.get_devices()
digit = TactileDigit.get_devices()

# Continuous devices: GoPro and AudioInterface
gopro_cameras = GoPro.get_devices()
audio_ifaces = AudioInterface.get_specific_devices(iface_name)

# If you do not know iface_name, you can search for it through
AudioInterface.print_found_devices()
```

**Handling lag in the teleoperation**:

Using slow discrete devices in the data collection loop can lead to a lag when teleoperating. Therefore, you might want to collect frames from discrete cameras asynchronously to the robot. This is handled in the `AsynchronousCamera` class. It acts as a `ContinuousCamera` wrapper for any `DiscreteCamera` that runs in a separate process:

```python
from real_robot_env.robot.hardware_cameras import AsynchronousCamera

# following example with a DepthAI camera:
async_cam = AsynchronousCamera[DepthAI](
    camera_class=DepthAI,
    # following parameters are from the DepthAI camera class
    device_id='18443010A1A7701200',
    name='top_cam',
    height=512,
    width=512,
    camera_type=DAICameraType.OAK_D_LITE
)
```

### Saved data

The following data is saved:

- Joint position and velocity for all leader and follower robots
- End effector position for all leader and follower robots
  - Has dimension 7, where first 3 is Cartesian position and last 4 is quaternion
- End effector velocity for all leader and follower robots
  - Has dimension 6, where first 3 is Cartesian velocity and last 3 is rotational velocity
- Gripper state of leader and follower robot
  - The state is binary (open or closed)
  - The threshold is set as max_width / 2
  - The state is -1 when closed and 1 when open
- A list of the timestamps at which data was collected
- Frames from each *discrete device*
- Full recordings from each *continuous device*

Note that some robots, e.g. Gello, do not provide all data categories. In such cases, the missing data categories are simply ignored.

### Replay collected data

- Using joint states: ``` python replay_data.py ```
    - Set `d` as the data directory to be replayed
- Using end effector states: ``` python replay_data_ee.py ```
    - Set `d` as the data directory to be replayed
