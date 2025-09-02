#!/bin/bash

echo -n "Password: "
read -s password
echo

sh $(dirname "$0")/tmux_grids.sh 4 2 robot_servers


scripts=(
    "launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=172.16.1.2 port=1234"
    "launch_gripper.py gripper=franka_hand gripper.executable_cfg.robot_ip=172.16.1.2 port=1235"
    "launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=172.16.2.2 port=4321"
    "launch_gripper.py gripper=franka_hand gripper.executable_cfg.robot_ip=172.16.2.2 port=4322"
    "launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=172.16.3.2 port=50051"
    "launch_gripper.py gripper=franka_hand gripper.executable_cfg.robot_ip=172.16.3.2 port=50052"
    "launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=172.16.4.2 port=50053"
    "launch_gripper.py gripper=franka_hand gripper.executable_cfg.robot_ip=172.16.4.2 port=50054"
)

for i in "${!scripts[@]}"
do
    tmux select-pane -t $i

    if ((i < 4))
    then
        tmux send "sshpass -p '$password' ssh alr_admin@141.3.53.152" ENTER
    else 
        tmux send "sshpass -p '$password' ssh alr_admin@141.3.53.154" ENTER
    fi

    tmux send "pkill -9 run_server" ENTER
    tmux send "mamba activate polymetis" ENTER
    tmux send "${scripts[i]}" ENTER 
done

tmux a