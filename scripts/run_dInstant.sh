#! /bin/bash

exp_name=SAM_HZ0201all
#exp_name=YLObj
workspace_name=SAM_HZ2_deps_all
data_path=data2/${exp_name}/
workspace_path=HZWJ/${workspace_name}
checkpoint_path=${workspace_path}/checkpoints/ngp_ep0050.pth

gui=0
test=0

if test ${gui} -gt 0
then
  python hz_nerf.py ${data_path} --workspace ${workspace_path}  \
  --HZ --depth_supervise --bound 1.0 --scale 0.25 --dt_gamma 0.02   \
  --ckpt ${checkpoint_path}   \
  --gui
elif test ${test} -gt 0
then
  python hz_nerf.py ${data_path} --workspace ${workspace_path}  \
  --HZ --depth_supervise --bound 1.0 --scale 0.25 --dt_gamma 0.02   \
  --ckpt ${checkpoint_path}   \
  --test
else
  python hz_nerf.py ${data_path} --workspace ${workspace_path}  \
  --HZ --bound 1.0 --scale 0.25 --dt_gamma 0.02   \
  --depth_supervise
#  --hz_train --depth_supervise
  # --depth_supervise  # --hz_train # \
#  --ckpt ${checkpoint_path}

fi
  echo Done!

#python hz_nerf.py ${data_path} --workspace ${workspace_path} --HZ --depth_supervise --bound 1.0 --scale 0.25 --dt_gamma 0.02
#python hz_nerf.py ${data_path} --workspace ${workspace_path} -O --bound 1.0 --scale 0.25 --dt_gamma 0.02
#python hz_nerf.py ${data_path} --workspace ${workspace_path} --HZ --depth_supervise --bound 1.0 --scale 0.25 --dt_gamma 0.02 --gui
#python hz_nerf.py ${data_path} --workspace ${workspace_path}  \
#  --HZ --depth_supervise --bound 1.0 --scale 0.25 --dt_gamma 0.02   \
#  --ckpt ${checkpoint_path}   \
#  --gui

