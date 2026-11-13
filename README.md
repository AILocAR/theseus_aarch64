# Theseus for ARM64/AArch64 CPU
Meta only provides pre-built Theseus wheels for x86 platforms. Therefore, Theseus must be built from source to run on ARM64/AArch64 platforms such as the NVIDIA GH200 or Apple Silicon.

For more details on Theseus, kindly refer to the original [README](https://github.com/AILocAR/theseus_aarch64/blob/main/README_original.md) and its [official webpage](https://sites.google.com/view/theseus-ai/).

##  Build from Source for ARM64/AArch64
The example provided here uses Python 3.10 and CUDA 13.0. The computational platform is NVIDIA GH200 Grace Hopper Superchip. For Apple Silicon, details are provided at the end.

### Create your software environment
````bash
conda create -p YOUR_ENVIRONMENT_PATH python=3.10
````

### Activate conda environment
````bash
conda activate YOUR_ENVIRONMENT_PATH
````

### Install PyTorch
````bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
````

### Install SuiteSparse
````bash
conda install -c conda-forge suitesparse
````

### Install scikit-sparse
Refer to this [GitHub page](https://github.com/AILocAR/scikit-sparse-aarch64)

### Download source code and build it
````bash
git clone https://github.com/AILocAR/theseus_aarch64.git
cd YOUR_PROJECT_DIRECTORY/theseus
python -m pip install -e . --no-build-isolation
````
## Build from Source for Apple Silicon
```bash
conda create -n YOUR_ENVIRONMENT_NAME python=3.10 -y
conda activate YOUR_ENVIRONMENT_NAME
conda install -c conda-forge pytorch -y                      # Make sure Pytorch is installed from conda-forge
conda install -c conda-forge suitesparse "scikit-sparse>=0.4.11,<0.5" -y
git clone https://github.com/facebookresearch/theseus.git
cd theseus
pip install -e . --no-build-isolation
python examples/simple_example.py      # Verification
```
