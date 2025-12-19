#!/bin/bash

# Panda Server Launch Script
# Run this on the NUC connected to the Franka robot

ROBOT_IP="${ROBOT_IP:-192.168.1.107}"
PORT="${PORT:-5556}"
GRIPPER_PORT="${GRIPPER_PORT:-5558}"
CONDA_ENV="${CONDA_ENV:-panda_server}"

echo "=========================================="
echo "Panda Server Launcher"
echo "=========================================="
echo "Robot IP:     $ROBOT_IP"
echo "State Port:   $PORT"
echo "Command Port: $((PORT + 1))"
echo "Gripper Port: $GRIPPER_PORT"
echo "=========================================="

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment '$CONDA_ENV'"
    echo "Create it with: conda env create -f environment.yaml"
    exit 1
fi

echo "Conda environment '$CONDA_ENV' activated"
echo "Starting server..."
echo ""

# Run the server
python panda_server.py \
    --robot-ip "$ROBOT_IP" \
    --port "$PORT" \
    --gripper-port "$GRIPPER_PORT"

