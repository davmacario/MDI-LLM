# Guide: setting up testing environment - Nvidia Jetson TX2

Since the JetPack 4.6.4, the latest version officially supporting Nvidia Jetson TX2s, only comes with Python 3.6 and Torch 1.10, and since some dependencies of this project require at least Python 3.8 (Huggingface Transformers, Tiktoken), it is necessary to perform some operations before being able to run the programs.

## Installing Python 3.8

Python 3.8 can be installed using the APT package manager:

```bash
sudo apt update
sudo apt install python3.8 python3.8-pip python3.8-venv -y
```

## Installing Torch 1.12

As said above, JetPack 4.6.4 only comes with Torch 1.10, despite the Jetsons being, in theory, able to support up to Torch 1.12 (with CUDA support - recent versions can be installed, but they will run on the CPU cores).
This is due to the fact that Torch 1.12 only supports Python versions 3.8 and above.

Having installed Python 3.8 it is possible to use Torch 1.12.
It is required, however, to compile this specific version from source, since no pre-compiled wheels exist for Jetson Tx2s.

**Optional**: install JTOP to monitor the stats, and activate all cores of the devices (some may be disabled by default)

```bash
sudo pip3 install -U jetson-stats
sudo systemctl restart jtop.service
sudo reboot
```

To use JTOP:

```bash
jtop
```

If some cores appear disabled, run the following command to activate them:

```bash
sudo nvpmodel -m 0  # Set mode to '0': all cores used
```

Then, edit the file `/boot/extlinux/extlinux.conf` and set `isolcpus=1-2` to `isolcpus=`.
Upon rebooting, all cores should be enabled by default.

Then, set up the environment for compiling Torch 1.12 (following [this guide](https://qengineering.eu/install-pytorch-on-jetson-nano.html)).

Setting up Torch compilation and install dependencies:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt-get install ninja-build git cmake
sudo apt-get install libjpeg-dev libopenmpi-dev libomp-dev ccache
sudo apt-get install libopenblas-dev libblas-dev libeigen3-dev
echo "export OPENBLAS_CORETYPE=ARMV8" >> "$HOME/.bashrc"
source "$HOME/.bashrc"
```

Create and activate the virtual environment

```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install future
pip3 install wheel mock pillow
pip3 install testresources
pip3 install setuptools==58.3.0
pip3 install Cython
pip3 install https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5+nv22.01-cp36-cp36m-linux_aarch64.whl
```

Clone the Torch repository from Github, selecting the branch for version 1.12:

```bash
cd "$HOME"
git clone -b v1.12.0 --depth=1 --recursive https://github.com/pytorch/pytorch.git
cd pytorch
pip3 install -r requirements.txt
sudo apt-get install clang-8
sudo ln -s /usr/bin/clang-8 /usr/bin/clang
sudo ln -s /usr/bin/clang++-8 /usr/bin/clang++
```

Then, proceed with the modifications described at the link, and launch the compiler.
It is suggested to compile inside a [tmux](https://www.redhat.com/sysadmin/introduction-tmux-linux) session, and also to run the following commands to invoke Jetson Clocks and fix the core frequencies to their maximum:

```bash
sudo jetson_clocks --store  # Store current "normal" state in the default path
sudo jetson_clocks  # Set the core clocks at the max frequency (2GHz)
sudo jetson_clocks --fan  # Ramp up the fan at 100%
```

Then, once the compilation is complete, run `sudo jetson_clocks --restore` to revert to the original settings.

Then, compile torch:

```bash
# set NINJA parameters
cd pytorch
export BUILD_CAFFE2_OPS=OFF
export USE_FBGEMM=OFF
export USE_FAKELOWP=OFF
export BUILD_TEST=OFF
export USE_MKLDNN=OFF
export USE_NNPACK=OFF
export USE_XNNPACK=OFF
export USE_QNNPACK=OFF
export USE_PYTORCH_QNNPACK=OFF
export USE_CUDA=ON
export USE_CUDNN=ON
export TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2"
export USE_NCCL=OFF
export USE_SYSTEM_NCCL=OFF
export USE_OPENCV=OFF
export MAX_JOBS=4
# set path to ccache
export PATH=/usr/lib/ccache:$PATH
# set clang compiler
export CC=clang
export CXX=clang++
# set cuda compiler
export CUDACXX=/usr/local/cuda/bin/nvcc
# create symlink to cublas
sudo ln -s /usr/lib/aarch64-linux-gnu/libcublas.so /usr/local/cuda/lib64/libcublas.so
# clean up the previous build, if necessary
python3 setup.py clean
# start the build
python3 setup.py bdist_wheel
```

This will produce a Python 3.8 wheel containing Torch 1.12 compiled to support CUDA on Jetson TX2s (and Jetson Nanos) - the name should be something like `torch-1.12.0a0+git67ece03-cp38-cp38-linux_aarch64.whl`.
To install the package, run (from within the Python 3.8 virtual environment):

```bash
cd dist
ls
sudo -H pip3 install torch-1.12.0a0+git67ece03-cp38-cp38-linux_aarch64.whl
```
