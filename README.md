# Robo3D-Central Host Part
This repository is some demo code from the Robo3D project, which only includes some code of the Central Host, while the robot's code is not included.

## Project:

This project explores the exciting field of 3D scene reconstruction from 2D images, with a focus on active mapping and planning using implicit representations. Leveraging the power of Neural Radiance Fields (NeRF), we've developed an online 3D reconstruction and planning system for active vision tasks.

NeRF has shown tremendous potential in photorealistic rendering and offline 3D reconstruction. Recently, it's also been used for online reconstruction and localization, forming the backbone of implicit SLAM systems. However, its use in active vision tasks, particularly active mapping and planning, is largely unexplored.

In this project, we introduce a novel RGB-D active vision framework that utilizes the radiance field representation for active 3D reconstruction and planning. This framework is implemented as an iterative dual-stage optimization problem, with alternating optimization for the radiance field representation and path planning.

We extend the existing work in several significant ways:

- **Depth Supervision**: We incorporate depth supervision into our model to further refine the 3D reconstruction process. This addition helps the model generate more accurate and detailed 3D structures.

- **Active Policy**: We introduce more active policy to our framework to guide the path planning process. The policy is designed to optimize for effective exploration and data collection in the 3D environment.

- **Robot's Reachable Space Model**: We incorporate a model of the robot's reachable space into our system. This model helps in path planning by taking into account the robot's physical constraints when exploring the environment.

Our experimental results indicate that our proposed method performs competitively with existing offline methods and even surpasses active reconstruction methods using NeRFs. We believe our project will contribute to the ongoing research in active 3D reconstruction and planning and open up new possibilities for real-world applications.

For one sentence summarization, our training and reconstructing process only spends less than 10 minutes, while the original version of NeRF spends hours or even tens of hours. 

## File Structure
```
├── nerf
│   ├── gui.py
│   ├── clip_utils.py
│   ├── network.py      # NeRFNetwork class
│   ├── provider.py     # Dataset provider
│   ├── renderer.py     # NeRFRenderer class for rendering the rays
│   ├── utils.py        # Main body, including Trainers and test Meters
├── scripts
│   ├── colmap2nerf.py  # Convert COLMAP poses to NeRF poses
│   ├── run_COLMAP.sh   # Script to run COLMAP
│   ├── run_dInstant.sh # Script to start training or testing
├── environment.yml
├── hz_nerf.py          # Entry script. 
├── requirements.txt
└── LICENSE
```

## Installation
```bash
git clone --recursive https://github.com/ZhouHaonanShens/Robo3D-A_Demo.git
cd Robo3D-A_Demo
```

### Install with pip
```bash
pip install -r requirements.txt
```

### Install with conda
```bash
conda env create -f environment.yml
conda activate Robo3D
```

## Usage
Run with only 2D images dataset
```bash
python hz_nerf.py ${data_path} --workspace ${workspace_path}  --HZ --bound 1.0 --scale 0.25
```

Run with 2D images and depths (KL Divergence)
```bash
python hz_nerf.py ${data_path} --workspace ${workspace_path}  --HZ --bound 1.0 --scale 0.25 --depth_supervise
```

Run with 2D images and depths (L2 Loss)
```bash
python hz_nerf.py ${data_path} --workspace ${workspace_path}  --HZ --bound 1.0 --scale 0.25 --depth_supervise_E2
```
