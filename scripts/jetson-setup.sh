#!/bin/bash

echo "Updating system"
sudo apt update
sudo apt upgrade -y

echo "Installing requirements"
PACKAGES=(
    "git"
    "vim"
    "htop"
    "neofetch"
    "curl"
    "wget"
    "tmux"
    "libomp5"
    "libopenblas-dev"
    "nvidia-jetpack"
    "python3.8"
    "python3.8-venv"
    "python3-pip"
)
for p in "${PACKAGES[@]}"; do
    sudo apt install "$p" -y
done
python3.8 -m pip install --upgrade pip
sudo pip3 install -U jetson-stats

echo "Installing tailscale (needs to be set up afterwards)"
curl -fsSL https://tailscale.com/install.sh | sh
