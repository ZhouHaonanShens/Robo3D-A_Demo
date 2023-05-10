#! /bin/bash

checkpoint_path=HZWorkspace/HZ0211mk_hz/checkpoints/ngp_ep0100.pth

#python main_nerf.py data/HZ0211m_mk/ --workspace HZWorkspace/HZ0211mask -O --bound 1.0 --scale 0.25 --dt_gamma 0.02
python hz_nerf.py data/HZ0211m_mk/ --workspace HZWorkspace/HZ0211mk_hzs --HZ --bound 1.0 --scale 0.25 --dt_gamma 0.02 \
#  --ckpt ${checkpoint_path}

