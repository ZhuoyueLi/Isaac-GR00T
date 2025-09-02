from real_robot_env.robot.hardware_franka import FrankaArm, ControlType
from real_robot_env.robot.hardware_frankahand import FrankaHand
from pathlib import Path
import torch
import time


data_dir = Path("/home/kkuryshev/audio-pipeline/data/simple_test/")#Path(__file__).parent / "data"
try:
    d = sorted([d for d in data_dir.iterdir()])[-1]
except IndexError as e:
    print("There is no collected data!")
    exit()
    

joint_pos = torch.load(d / "201 leader/joint_pos.pt")
joint_vel = torch.load(d / "201 leader/joint_vel.pt")
gripper_command = torch.load(d / "201 leader/gripper_state.pt")

delta_t = 0.05 #0.034 too fast

p4 = FrankaArm(name='p4', ip_address='141.3.53.63', port=50053, control_type=ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL)
assert p4.connect(), f"Connection to {p4.name} failed"

p4_hand = FrankaHand(name="p4_hand", ip_address='141.3.53.63', port=50054)
assert p4_hand.connect(), f"Connection to {p4_hand.name} failed"

for i in range(joint_pos.shape[0]):
    p4.apply_commands(q_desired=joint_pos[i], qd_desired=joint_vel[i])
    p4_hand.apply_commands(width=gripper_command[i])
    time.sleep(delta_t) 

p4.close()
p4_hand.close()