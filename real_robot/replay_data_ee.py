from real_robot_env.robot.hardware_franka import FrankaArm, ControlType
from real_robot_env.robot.hardware_frankahand import FrankaHand
from pathlib import Path
import torch
import time


data_dir = Path(__file__).parent / "data"
try:
    d = sorted([d for d in data_dir.iterdir()])[-1]
except IndexError as e:
    print("There is no collected data!")
    exit()

ee_pos = torch.load(d / "leader_ee_pos.pt")
ee_vel = torch.load(d / "leader_ee_vel.pt")
gripper_command = torch.load(d / "leader_gripper_state.pt")

delta_t = 0.034

p4 = FrankaArm(name='p4', ip_address='141.3.53.154', port=50053, control_type=ControlType.CARTESIAN_IMPEDANCE_CONTROL)
assert p4.connect(), f"Connection to {p4.name} failed"

p4_hand = FrankaHand(name="p4_hand", ip_address='141.3.53.154', port=50054)
assert p4_hand.connect(), f"Connection to {p4_hand.name} failed"

for i in range(ee_pos.shape[0]):
    p4.robot.update_current_policy({
        "ee_pos_desired": ee_pos[i][:3],
        "ee_quat_desired": ee_pos[i][3:],
        "ee_vel_desired": ee_vel[i][:3],
        "ee_rvel_desired": ee_vel[i][3:]
    })
    p4_hand.apply_commands(width=gripper_command[i])
    time.sleep(delta_t) 

p4.close()
p4_hand.close()