#!/bin/bash

echo -n "Password: "
read -s password
echo

sh $(dirname "$0")/tmux_grids.sh 2 2 fci

for i in {0..3}
do
    tmux select-pane -t $i

    if ((i < 2))
    then
        tmux send "sshpass -p '$password' ssh alr_admin@141.3.53.152" ENTER
    else 
        tmux send "sshpass -p '$password' ssh alr_admin@141.3.53.154" ENTER
    fi

    tmux send "python ~/robot/franka_lock_unlock/__init__.py -u -c -p 172.16.$(($i + 1)).2 a azsxdcfv1" ENTER
done

tmux a