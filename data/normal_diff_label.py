import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytlsd
import h5py
import os
import pickle as pk
import pandas as pd

from tqdm import tqdm
from bresenham import bresenham
from joblib import Parallel, delayed

def pix_diff(image, step_size):
    # Create a padded version of the image to handle edge cases
    padded_image = np.pad(image, ((step_size, step_size), (step_size, step_size), (0, 0)), mode='edge')

    # Initialize the output image with the same size as the input image
    res_image = np.zeros((image.shape[0], image.shape[1]))

    delta_ps = []
    for x in range(-step_size, step_size+1):
        for y in range(-step_size, step_size+1):
            if abs(x) + abs(y) >= step_size:
                delta_ps.append((x, y))

    # Iterate over all the pixels in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            cx = i + step_size
            cy = j + step_size
            
            normal_diffs = []
            for t in range(len(delta_ps) // 2):
                x1, y1 = delta_ps[t]
                x2, y2 = delta_ps[-t]
                n1 = padded_image[cx + x1, cy + y1]
                n2 = padded_image[cx + x2, cy + y2]

                normal_diffs.append(np.dot(n1, n2))

            # Set the pixel value in the output image to the variance
            res_image[i, j] = np.mean(normal_diffs)

    return res_image

def process(split, scene_name, cam_name, frame_id, skip_exists = True):
    res_folder = os.path.join(split, scene_name, 'normals_diff', cam_name)
    lineseg_folder = os.path.join(split, scene_name, 'lines', cam_name, 'segments')
    linepreview_folder = os.path.join(split, scene_name, 'lines', cam_name, 'preview')
    # res_folder = os.path.join(split, scene_name, 'omnidata_normals_diff', cam_name)
    # linepreview_folder = os.path.join(split, scene_name, 'lines', cam_name, 'preview_omnidata_normals')
    os.makedirs(res_folder, exist_ok=True)
    os.makedirs(lineseg_folder, exist_ok=True)
    os.makedirs(linepreview_folder, exist_ok=True)

    linesegment_file = os.path.join(lineseg_folder, 'frame.{:04d}.segments.pk'.format(frame_id))
    linepreview_file = os.path.join(linepreview_folder, 'frame.{:04d}.preview.png'.format(frame_id))
    normal_diff_file = os.path.join(res_folder, 'frame.{:04d}.npz'.format(frame_id))

    # skip existed
    if skip_exists and os.path.exists(linepreview_file) and os.path.exists(linesegment_file) and os.path.exists(normal_diff_file):
        return

    base_dir = os.path.join(split, scene_name, 'images')
    preview_file = os.path.join(base_dir, f'scene_{cam_name}_final_preview', 'frame.{:04d}.color.jpg'.format(frame_id))
    
    img = cv2.imread(preview_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flt_img = gray.astype(np.float64)

    segments = pytlsd.lsd(flt_img)
    data = {'segments': segments}

    for segment in segments:
        cv2.line(img, (int(segment[0]), int(segment[1])), (int(segment[2]), int(segment[3])), (255, 0, 0))

    normal_file = os.path.join(base_dir, f'scene_{cam_name}_geometry_hdf5', 'frame.{:04d}.normal_cam.hdf5'.format(frame_id))
    f = h5py.File(normal_file, 'r')
    normal_data = f['dataset'][:]
    normal_data = np.divide(normal_data, np.linalg.norm(normal_data, axis=-1, keepdims=True))

    # normal_file = os.path.join(base_dir, f'scene_{cam_name}_omnidata_pred', 'frame.{:04d}.normal.npz'.format(frame_id))
    # normal_data = np.load(normal_file)['normals']
    # normal_data = cv2.resize(normal_data, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # normal_data = np.divide(normal_data, np.linalg.norm(normal_data, axis=-1, keepdims=True))

    normal_diff = pix_diff(normal_data, 2)

    flags = []
    for i, seg in enumerate(segments):
        seg2d = seg[:4].astype(int)
        seg_pts = np.array(list(bresenham(seg2d[0], seg2d[1], seg2d[2], seg2d[3])))
        seg_pts = seg_pts[(seg_pts[:, 0] >= 0) * (seg_pts[:, 1] >= 0) * (seg_pts[:, 0] < gray.shape[1]) * (seg_pts[:, 1] < gray.shape[0])]
        v = np.median(normal_diff[seg_pts[:,1], seg_pts[:,0]])
        if v < np.cos(np.pi / 4):
            flags.append(True)
        else:
            flags.append(False)

    data['structure'] = flags
    for f, segment in zip(flags, segments):
        if not f:
            continue
        cv2.line(img, (int(segment[0]), int(segment[1])), (int(segment[2]), int(segment[3])), (0, 0, 255))

    with open(linesegment_file, 'wb') as f:
        pk.dump(data, f)
    cv2.imwrite(linepreview_file, img)
    np.savez_compressed(normal_diff_file, normals_diff=normal_diff)

if __name__ == '__main__':
    df = pd.read_csv('metadata_images_split_scene_v1.csv')
    df = df[df['scene_name'] < 'ai_004_001']
    rows = []
    for i, row in tqdm(df.iterrows()):
        if not row.included_in_public_release:
            continue
        rows.append(row)
    
    Parallel(n_jobs=6)(delayed(process)(row.split_partition_name, row.scene_name, row.camera_name, row.frame_id) for row in tqdm(rows))
    # for row in tqdm(rows):
    #     split = row.split_partition_name
    #     scene = row.scene_name
    #     cam_name = row.camera_name
    #     frame_id = row.frame_id
    #     process(split, scene, cam_name, frame_id)

