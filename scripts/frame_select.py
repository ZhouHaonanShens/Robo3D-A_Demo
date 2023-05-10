import json
import os
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--exp_name', type=str, default='HZ0201all')
parser.add_argument('--exp_name', type=str, default='YLObj')
parser.add_argument('--divide_zone', action='store_true')
parser.add_argument('--early_skip', type=int, default=15)
parser.add_argument('--num_per_zone', type=int, default=12)
parser.add_argument('--num_zone', type=int, default=12)
opt = parser.parse_args()

# entropy = {
#     'max_idxs': [10, 8, 11, 8, 0, 0, 9, 11, 11, 2, 0, 2],
#     'max_value': [3.7617953, 3.7963014, 4.113614, 4.2246633, 4.13454, 3.8208928, 3.608536, 3.7555687, 4.287876, 4.2935295, 3.9340706, 3.7861404]
# }  # the selected frames without depth supervision

entropy = {
    # 'max_idxs': [6, 9, 1, 8, 11, 3, 7, 9, 1, 9, 0, 3],
    'max_idxs': [],
    'max_value': [5.5889063, 5.524664, 5.520462, 5.4975495, 5.528879, 5.535035, 5.508785, 5.4266667, 5.452477, 5.482142, 5.486807, 5.4867897]
}  # the selected frames Second with mask R45

# entropy = {
#     'max_idxs': [11, 11, 4, 0, 2, 9, 11, 0, 1, 11, 6, 0],
#     'max_value': [2.5003068, 2.7346573, 2.911503, 2.742849, 2.494193, 2.2208903, 2.6728292, 2.824918, 2.508811, 3.1094964, 3.3192735, 3.0167086]
# }  # the selected frames Second with mask

data_path = os.path.join('../data2', opt.exp_name, 'transforms_train_all.json')

json_file = 'transforms_train_all.json' if os.path.isfile(data_path) else 'transforms_train.json'

data_path = os.path.join('../data2', opt.exp_name, json_file)
fp = open(data_path, 'r')
data = json.load(fp)
fp.close()

max_idxs = entropy['max_idxs']
zone_num = len(max_idxs)
early_skip = opt.early_skip
selected_frames = data['frames'][:early_skip]


for i in range(zone_num):
    selected_idx = early_skip + opt.num_per_zone*i + max_idxs[i]
    selected_frames.append(data['frames'][selected_idx])
    ff = np.array(data['frames'][selected_idx]['transform_matrix'])

radius = np.zeros(len(selected_frames))
for i in range(len(selected_frames)):
    ff = np.array(selected_frames[i]['transform_matrix'])
    radius[i] = np.sqrt(ff[0, 3]**2 + ff[1, 3]**2 + ff[2, 3])

print('Mean Radius is:', np.mean(radius) * 0.25)

selected_data = deepcopy(data)
selected_data['frames'] = selected_frames

with open(os.path.join('../data2', opt.exp_name, 'transforms_train.json'), 'w') as fp:
    json.dump(selected_data, fp=fp, sort_keys=False, indent=2)
with open(os.path.join('../data2', opt.exp_name, 'transforms_train_all.json'), 'w') as fp:
    json.dump(data, fp=fp, sort_keys=False, indent=2)


