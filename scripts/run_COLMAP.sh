#! /bin/bash

exp_name=HZL0314_55_2

divide_zone=0
run_colmap=1

# [INFO] avg camera distance from origin 4.63676202881754

if test ${run_colmap} -gt 0
then
  image_path=./data2/${exp_name}/images
  python scripts/colmap2nerf.py --images ${image_path} --hold 0  --run_colmap --run_colmap_dense
fi
echo COLMAP Part Finished!

if test ${divide_zone} -gt 0
then
  python scripts/data_split.py --exp_name ${exp_name} --divide_zone
else
  python scripts/data_split.py --exp_name ${exp_name}
fi
echo Data Split Part Finished!
