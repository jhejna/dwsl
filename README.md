# Distance Weighted Supervised Learning (DWSL)

This is the official codebase for Distance Weighted Supervised Learning (DWSL) by Joey Hejna, Jensen Gao, and Dorsa Sadigh. This repository contains code for dwsl, all baselines, environments, and datasets. Our implementations are based on [research-lightning](https://github.com/jhejna/research-lightning). For more details on some of the advanced features in the repisotry, see the research-lightning repo.

In order to run all the experiments, one must download or create the datasets.

## Installation

Warning: this repository is currently setup to use experimental pytorch 2.0 features.

Here are the following steps to setup the base repository:
1. Clone the repository to your desired location using `git clone`.
2. Create the conda environment using `conda env create -f environment_<cpu or gpu>.yaml`.
3. Install the research package via `pip install -e research`.
4. Modify the `setup_shell.sh` script by updating the appropriate values as needed. The `setup_shell.sh` script should load the environment, move the shell to the repository directory, and additionally setup any external dependencies. All the required flags should be at the top of the file.

When using the repository, you should be able to setup the environment by running `. path/to/setup_shell.sh`.

In order to run experiments for different environments and datasets, we need to set them up individually. The antmaze tasks are already handled by d4rl.

### Gym Robotics

For the Fetch environments, we create datasets by training oracle policies with TD3+HER, then collecting random data or noised demonstrations. For each desired environment, repeat the instructions below with a different environment name.

1. Train an oracle policy:
```
python scripts/train.py --config configs/datasets/td3_her_fetch_pick.yaml --path path/to/output
```
2. Collect a state dataset using
```
python scripts/create_dataset.py --path dataset/output/path --num-ep 2000 --noise 2.0 --random-percent 0.0 --checkpoint path/to/model/best_model.pt
```
3. Collect an image dataset using the conversion script
```
python scripts/create_fetch_image_dataset.py --path dataset/output/path --num-ep 2000 --noise 2.0 --random-percent 0.0 --checkpoint path/to/model/best_model.pt
```

Our methods are also compatible with the datasets from WGCSL. We use the processed versions of the datasets from [GoFar]() found [here](https://drive.google.com/file/d/1niq6bK262segc7qZh8m5RRaFNygEXoBR/view). These datasets can be used by replacing the `HindsightReplayBuffer` dataset class in the configs with the `WGCSLDataset`.

For the hand environment, we direclty used the above datasets. We rendered them to images for pixel experiments. Note that we also needed to fix a rendering bug in the mujoco file. To use the image versions of the Hand environments, replace the hand environment assets in your gym installation with the one found here: `https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/assets/hand/reach.xml`. This fixes the rendering bug documented [here](https://github.com/openai/gym/issues/2061). We recommend using `wget`  to download the file, and cp-ing it into the gym installation, whose location can by found by printing `gym.__file__`.

Then, make sure to download the WGCSL datasets for running the hand experiments. To construct the image datasets, use the following script:
```
python scripts/create_hand_image_dataset.py --path path/to/hand/dataset --output output/storage/directory
```

### Franka Relay Kitchen

For the Franka Kitchen tasks, we use the datasets from [Conditional Behavior Transformers](https://github.com/jeffacce/play-to-policy).

1. Clone the relay policy learning repository `git clone https://github.com/google-research/relay-policy-learning`.
2. Add the adept envs to your python path via `export PYTHONPATH=$PYTHONPATH:$/path/to/relay-policy-learning/adept_envs`. We add this to `setup_shell.sh`.
3. Download the datasets [here](https://osf.io/q3dx2/) and unzip them
4. Update the kitchen configs to point to the correct dataset path.

### Robomimic Tasks
In order to run the robomimic experiments, you need to install the [robomimic](https://robomimic.github.io/docs/introduction/installation.html) package and the [robosuite](https://robosuite.ai/) package. We install these dependencies in the following manner:

Robosuite:
1. Git clone the robosuite repository, found [here](https://github.com/ARISE-Initiative/robosuite).
2. Checkout the `offline_study` branch
3. install the package to the conda environment without dependencies via `pip install -e . --no-dependencies`.
4. Relevant as of 2/14/2023: Robosuite has not updated their package to be compatible with Python 3.10. Change `from collections import Iterable` to `from collections.abc import Iterable` in `robosuite/models/arenas/multi_table_arena.py` and `robosuite/utils/placement_samplers.py`.
5. Install `numba` via `conda install numba`.

Robomimic:
1. Git clone the robomimic repository, found [here](https://github.com/ARISE-Initiative/robosuite).
2. install the package to the conda environment without dependencies via `pip install -e . --no-dependencies`.
3. Download the datasets per the instructions [here](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html)

Finally, make sure to edit the robomimic configs correctly point to the download locations of the dataset.

## Usage

You should be able to activate the development enviornment by running `. path/to/setup_shell.sh`. Experiments can be run one at a time with the following command:
```
python scripts/train.py --config path/to/config --path path/to/storage/directory
```

This repository also supports running multiple jobs via a launcher. This makes use of the `.json` files that can be found in some of the config directories. This can be done for slurm by:
```
python tools/run_slurm.py --slurm-arg-1 <slurm_arg_1> .. --slurm-arg-n <slurm_arg_n> --arguments config=path/to/config path=path/to/storage/directory
```
Or using a local sweeper via
```
python tools/run_slurm.py --sweeper-arg-1 <sweeper_arg_1> .. --sweeper-arg-n <sweeper_arg_n> --arguments config=path/to/config path=path/to/storage/directory
```
For more details on this, see the [research-lightning repository](https://github.com/jhejna/research-lightning)

## Attribution
This framework has an MIT license as found in the [LICENSE](LICENSE) file.

If you use this package, please cite our paper. Here is the associated Bibtex:
```
TODO
```
