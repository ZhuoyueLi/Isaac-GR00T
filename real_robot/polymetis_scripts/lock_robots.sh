#!/bin/bash

echo -n "Password: "
read -s password
echo

tmux kill-session -t robot_servers

for i in {0..3}
do
    if ((i < 2))
    then
        tmux send "sshpass -p '$password' ssh alr_admin@141.3.53.152" ENTER
    else 
        tmux send "sshpass -p '$password' ssh alr_admin@141.3.53.154" ENTER
    fi

    tmux send -t fci.$i C-c
    sleep 1
    tmux send -t fci.$i "python ~/robot/franka_lock_unlock/__init__.py 172.16.$(($i + 1)).2 a azsxdcfv1" ENTER
done

sleep 2
tmux kill-session -t fci

# tmux a