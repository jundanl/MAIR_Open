conda create -n MAIR python=3.9
conda activate MAIR
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm termcolor scikit-image imageio nvidia-ml-py3 h5py wandb opencv-python trimesh[easy] einops
pip uninstall numpy
pip install numpy<2