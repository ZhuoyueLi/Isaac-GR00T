#!/bin/sh
# --------------------------------------------------------------------------
#
# Simple script to generate a new tmux session in grid layout.
# All generated panes will have the same size.
# Usage: ./tmux_grid.sh {vertical panes} {horizontal panes}
#
# Author: Aggelos Stamatiou, July 2022
#
# This source code is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this source code. If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------------

# Auxiliary function to display usage message
usage() {
    echo "Usage: ./tmux_grid.sh {vertical panes} {horizontal panes}"
    exit 1
}

# Auxiliary function to validate input parameters
validate() {
    if [ -z "$1" ]; then
        echo "Vertical panes not provided."
        usage
    fi
    
    if [ -z "$2" ]; then
        echo "Horizontal panes not provided."
        usage
    fi
    
    if [ "$1" -le 0 ] || [ "$2" -le 0 ]; then
        echo "Input numbers must be positive."
        usage
    fi
}

# Validate input parameters
v=$1
h=$2
validate "$v" "$h"

session_name=$3

# Initialize tmux session
tmux new-session -d -s $session_name

# Horizontal split
h_bound=$(( h - 1 ))
if [ "$h" -gt 1 ]; then
    # Since tmux already creates a pane, we generate h-1
    for i in $(seq 1 $h_bound)
    do
        tmux split-window -h
    done
    # Adjust panes width
    tmux select-layout even-horizontal
fi

# Vertical split
if [ "$v" -gt 1 ]; then
    # Since we used the even-horizontal layout,
    # we have to split in vertical orientation each of
    # the generated panes, using the pane height in lines.
    tmux select-pane -t 0
    v_total=$(tmux display-message -p '#{pane_height}')
    v_size=$(( v_total /  v ))
    v_bound=$(( v - 1 ))
    for i in $(seq  0 $h_bound)
    do
        # Each time we generate a new pane, rest panes
        # index moves, so we need to always select the first
        # pane of the horizontal line we split.
        pane=$(( i * v ))
        tmux select-pane -t $pane
        for j in $(seq 1 $v_bound)
        do
            tmux split-window -l $v_size -v
            tmux select-pane -t $pane
        done
    done
    tmux select-pane -t 0
fi

# Splitting complete, starting tmux session
# tmux attach
