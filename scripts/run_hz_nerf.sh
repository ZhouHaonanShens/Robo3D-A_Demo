#! /bin/bash

exp_name=HZ0214_55all
workspace_name=HZ0217_DPT2
data_path=data/${exp_name}/
workspace_path=HZWorkspace/${workspace_name}
checkpoint_path=${workspace_path}/checkpoints/ngp_ep0050.pth

python hz_nerf.py ${data_path} --workspace ${workspace_path} --HZ --test_entropy \
  --bound 1.0 --scale 0.25 --dt_gamma 0.02  \
  --ckpt ${checkpoint_path}

