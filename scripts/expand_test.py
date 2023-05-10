import os
import json
from copy import deepcopy
import numpy as np
import argparse

from scipy.linalg import sqrtm
from scipy.spatial.transform import Rotation as R


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='HZ0201all')
    parser.add_argument('--divide_zone', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--early_skip', type=int, default=15)
    parser.add_argument('--num_per_zone', type=int, default=12)
    parser.add_argument('--num_zone', type=int, default=12)
    parser.add_argument('--order_insert', type=int, default=6)
    opt = parser.parse_args()
    np.set_printoptions(precision=4, suppress=True)

    test_original_path = os.path.join('../data', opt.exp_name, 'transforms_test_original.json')

    # json_file = 'transforms_test_original.json' if os.path.isfile(test_original_path) else 'transforms_test.json'
    json_file = 'transforms_train_all.json'

    data_path = os.path.join('../data', opt.exp_name, json_file)
    fp = open(data_path, 'r')
    data = json.load(fp)
    fp.close()

    frames = data['frames']
    if 'train' in json_file and len(frames) >= 15:
        frames = frames[:15]
    new_frames = []

    order_insert = 4  # opt.order_insert
    num_insert = 2**order_insert

    for i in range(len(frames) - 1):
        trans_from = np.array(frames[i]['transform_matrix'])
        trans_to = np.array(frames[i + 1]['transform_matrix'])
        R_from = trans_from[:3, :3]
        R_to = trans_to[:3, :3]
        T_diff = np.matmul(np.linalg.inv(trans_from), trans_to)
        print('\nnp.matmul(trans_from, T_diff): \n', np.matmul(trans_from, T_diff))
        print('\n trans_to:\n', trans_to)
        R_diff = R.from_matrix(T_diff[:3, :3])
        euler_diff = R_diff.as_euler('xyz', degrees=True)
        # euler_diff = R.from_matrix(R_to).as_euler('xyz', degrees=True) \
        #     - R.from_matrix(R_from).as_euler('xyz', degrees=True)
        position_diff = trans_to[:3, -1] - trans_from[:3, -1]
        print(euler_diff)

        euler_diff = euler_diff/num_insert
        position_diff = position_diff/num_insert
        print(euler_diff)

        scale = 1.0
        decay_rate = 0.05
        opt.scale = True
        if opt.scale:
            scale -= i*decay_rate
        f = deepcopy(frames[i])
        # new_frames.append(f)
        trans = deepcopy(trans_from)
        r_diff = R.from_euler('xyz', euler_diff, degrees=True).as_matrix()
        for j in range(num_insert):
            rate = float(j)/num_insert
            trans[:3, :3] = np.matmul(trans[:3, :3], r_diff)
            # trans[:3, -1] += position_diff
            trans[:3, -1] = (1-rate)*trans_from[:3, -1] + rate*trans_to[:3, -1]
            if opt.scale:
                trans[:3, -1] *= scale - (j/num_insert)*decay_rate
            f['transform_matrix'] = trans
            new_frames.append(deepcopy(f))
        print('\n trans_to:\n', trans_to)
        print('\n trans_final:\n', trans)

    if json_file == 'transforms_test.json':
        with open(os.path.join('../data', opt.exp_name, 'transforms_test_original.json'), 'w') as fp:
            json.dump(data, fp=fp, sort_keys=False, indent=2, cls=NpEncoder)
    data['frames'] = new_frames
    with open(os.path.join('../data', opt.exp_name, 'transforms_test.json'), 'w') as fp:
        json.dump(data, fp=fp, sort_keys=False, indent=2, cls=NpEncoder)




        # for _ in range(order_insert - 1):
        #     # print('R_diff original:\n', T_diff)
        #     T_diff = np.real(np.array(sqrtm(T_diff)))
        #     # print('R_diff sqrted square:', np.real(T_diff.dot(T_diff)), '\n\n')
        # f = deepcopy(frames[i])
        # new_frames.append(f)
        # trans = deepcopy(trans_from)
        # for _ in range(num_insert):
        #     trans = np.matmul(trans, T_diff)
        #     f['transform_matrix'] = trans
        #     new_frames.append(f)
        # print('\n final trans:\n', trans)
        # print('\n final plus trans:\n', trans.dot(T_diff))
        # print('\n trans_to:\n', trans_to)






    # device = 'cpu'

    # train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
    # test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
    # valid_loader = NeRFDataset(opt, device=device, type='val').dataloader()
    #
    # train_poses_np = train_loader._data.poses.numpy()
    # test_poses_np = test_loader._data.poses.numpy()
    # valid_poses_np = valid_loader._data.poses.numpy()

    # R_tests = []
    # for i in range(test_poses_np.shape[0]):
    #     R_test = R.from_matrix(test_poses_np[i, :3, :3]).as_euler('xyz', degrees=True)
    #     R_tests.append(R_test)
    # R_tests = np.array(R_tests)

    print("All Done!")

