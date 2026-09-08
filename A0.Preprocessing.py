import os
import glob
import nibabel as nib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
import json
import imageio
import matplotlib.pyplot as plt

'''

'''

save_basepath = os.path.join('./datasets/totalsegmentator_retrieval')
for i in ['image', 'seg_gt', 'seg_vista3d']:
    os.makedirs(os.path.join(save_basepath, f'train/{i}'), exist_ok=True)
    os.makedirs(os.path.join(save_basepath, f'gallery/{i}'), exist_ok=True)

len(glob.glob(os.path.join(save_basepath, f'gallery/image/*')))

# TotalSegmentatorV2_5_folds.json  train/ test split
rootpath = './datasets'
datasetname = 'totalsegmentator'


vol_base_path = os.path.join(rootpath, datasetname)
file_path = os.path.join(rootpath, datasetname, 'TotalSegmentatorV2_5_folds.json')
with open(file_path, 'r' ) as f:
    data = json.load(f)


_data_type = 'train' # train  gallery query

imgs = glob.glob(f'./datasets/totalsegmentator_retrieval/{_data_type}/image/*')
gal_list = np.array([ii.split('/')[-1].split('.')[0].replace('_slice', '').split('_') for ii in imgs])
data.keys()

data.keys()
len(data['testing'])
len(data['training'])

if _data_type == 'train':
    _data = data['training']
    sampled_slice_num = None
elif _data_type == 'gallery':
    _data = data['testing']
    sampled_slice_num = None
elif _data_type == 'query':
    _data = data['testing']
    sampled_slice_num = 10
else:
    print('select _data_type train  gallery query')

for i in tqdm(range(len(_data))):
    _vol = _data[i]
    img_vol_path = _vol['image']
    vol_name = os.path.split(img_vol_path)[0]
    gt_seg_vol_path  = _vol['image'].replace('ct.nii.gz', 'ct_gt_seg.nii.gz')
    vista3d_seg_vol_path  = _vol['image'].replace('ct.nii.gz', 'vista3d/ct/ct_seg.nii.gz')

    ct_volume = nib.load(os.path.join(vol_base_path, img_vol_path))
    ct_data = ct_volume.get_fdata()
    ct_shape  = ct_data.shape

    gt_seg_volume = nib.load(os.path.join(vol_base_path, gt_seg_vol_path))
    gt_seg_data = gt_seg_volume.get_fdata()
    gt_seg_shape = gt_seg_data.shape

    vista_seg_volume = nib.load(os.path.join(vol_base_path, vista3d_seg_vol_path))
    vista_seg_data = vista_seg_volume.get_fdata()
    vista_seg_shape = vista_seg_data.shape

    if not (ct_shape == gt_seg_shape and ct_shape == vista_seg_shape):
        print('volums shape are not equal')
        continue

    # plt.imshow(ct_data[:,:,20]); plt.show()
    # -------------------------
    # train의 경우 마스크가 있는 slice index 찾기
    # -------------------------
    mask_slices = []

    for z in range(gt_seg_data.shape[2]):
        if np.any(gt_seg_data[:, :, z] > 0):  # 해당 slice에 마스크가 하나라도 있으면
            mask_slices.append(z)

    mask_vista_slices = []
    for z in range(vista_seg_data.shape[2]):
        if np.any(vista_seg_data[:, :, z] > 0):   # 해당 slice에 마스크가 하나라도 있으면
            mask_vista_slices.append(z)
    _idx = np.isin(mask_slices, mask_vista_slices)
    mask_slices = np.array(mask_slices)[_idx]

    # -------------------------
    # interval=3 으로 샘플링
    # -------------------------

    if _data_type == 'train':
        sampled_slices = mask_slices[::3]
        print("sampled slices:", len(sampled_slices))
        # print("sampled first 10:", sampled_slices[:10])
    elif _data_type == 'gallery':
        sampled_slices = mask_slices[::4]
        print("sampled slices:", len(sampled_slices))
    else:
        _gal_idx = gal_list[gal_list[:, 0] == vol_name][:, 1].astype(int)
        mask_slices = mask_slices[~np.isin(mask_slices, _gal_idx)]

        # sampled_slices = mask_slices[::3]
        # print("sampled slices:", len(sampled_slices))
        if len(mask_slices) < sampled_slice_num:
            sampled_slice_num = len(mask_slices)
            print("warning [len(mask_slices) < sampled_slice_num]")
        sampled_slices = np.random.choice(mask_slices, sampled_slice_num)
    # -------------------------
    # sampled slice 저장
    # -------------------------
    for z in sampled_slices:
        img_slice = ct_data[:, :, z]
        # img_slice = normalize_ct(img_slice)

        gt_seg_slice = gt_seg_data[:, :, z]
        vista_seg_slice = vista_seg_data[:, :, z]

        save_path = os.path.join(save_basepath, f'{_data_type}/image', f"{vol_name}_slice_{z:04d}.npy")
        # imageio.imwrite(save_path, img_slice)
        np.save(save_path, img_slice)


        save_path = os.path.join(save_basepath, f'{_data_type}/seg_gt', f"{vol_name}_slice_{z:04d}.npy")
        np.save(save_path, gt_seg_slice)

        save_path = os.path.join(save_basepath, f'{_data_type}/seg_vista3d', f"{vol_name}_slice_{z:04d}.npy")
        np.save(save_path, vista_seg_slice)