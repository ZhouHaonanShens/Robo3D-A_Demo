import json
import os
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str)
parser.add_argument('--divide_zone', action='store_true')
parser.add_argument('--early_skip', type=int, default=15)
parser.add_argument('--num_per_zone', type=int, default=12)
parser.add_argument('--num_zone', type=int, default=12)
opt = parser.parse_args()

data_path = os.path.join('./data2', opt.exp_name, 'transforms_ready.json')

fp = open(data_path, 'r')
data = json.load(fp)
print('camera_angle_x:', data['camera_angle_x'])

train = deepcopy(data)
train['frames'] = []
test = deepcopy(train)
val = deepcopy(train)
zones = []
num_zone = opt.num_zone
for i in range(num_zone):
    zone = deepcopy(train)
    zones.append(zone)

train_idx = 0

for f in data['frames']:
    if 'train' in f['file_path']:
        train['frames'].append(f)
        idx = train_idx - opt.early_skip
        if opt.divide_zone and idx >= 0:
            zone_idx = int(idx / opt.num_per_zone)
            in_zone_idx = int(idx % opt.num_per_zone)
            if zone_idx >= num_zone:
                print("Num configuration wrong!")
            else:
                zones[zone_idx]['frames'].append(f)
        train_idx += 1
        continue
    elif 'test' in f['file_path']:
        test['frames'].append(f)
        continue
    elif 'val' in f['file_path']:
        val['frames'].append(f)
        continue
    else:
        print("file_path wrong")

train['frames'].sort(key=lambda d: d['file_path'])
test['frames'].sort(key=lambda d: d['file_path'])
val['frames'].sort(key=lambda d: d['file_path'])

with open(os.path.join('./data2', opt.exp_name, 'transforms_train.json'), 'w') as fp:
    json.dump(train, fp=fp, sort_keys=False, indent=2)
with open(os.path.join('./data2', opt.exp_name, 'transforms_test.json'), 'w') as fp:
    json.dump(test, fp=fp, sort_keys=False, indent=2)
with open(os.path.join('./data2', opt.exp_name, 'transforms_val.json'), 'w') as fp:
    json.dump(val, fp=fp, sort_keys=False, indent=2)

if opt.divide_zone:
    for i in range(num_zone):
        zone = zones[i]
        zone['frames'].sort(key=lambda d: d['file_path'])
        with open(os.path.join('./data2', opt.exp_name, f'transforms_zone{i:02d}.json'), 'w') as fp:
            json.dump(zone, fp=fp, sort_keys=False, indent=2)
