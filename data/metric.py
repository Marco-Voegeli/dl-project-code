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

tp, fp, tn, fn = 0, 0, 0, 0

def process(split, scene_name, cam_name, frame_id, skip_exists = True):
    lineseg_folder = os.path.join(split, scene_name, 'lines', cam_name, 'segments')
    pred_folder = os.path.join('data', split, scene_name, 'pred_normals_diff', cam_name)
    gt_folder = os.path.join(split, scene_name, 'normals_diff', cam_name)

    pred_file = os.path.join(pred_folder, 'frame.{:04d}.color.jpg.npz'.format(frame_id))
    gt_file = os.path.join(gt_folder, 'frame.{:04d}.npz'.format(frame_id))
    linesegment_file = os.path.join(lineseg_folder, 'frame.{:04d}.segments.pk'.format(frame_id))

    base_dir = os.path.join(split, scene_name, 'images')
    preview_file = os.path.join(base_dir, f'scene_{cam_name}_final_preview', 'frame.{:04d}.color.jpg'.format(frame_id))
    
    img = cv2.imread(preview_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flt_img = gray.astype(np.float64)

    with open(linesegment_file, 'rb') as f:
        segments = pk.load(f)['segments']

    preds = np.load(pred_file)['normals_diff']
    gts = np.load(gt_file)['normals_diff']

    global tp, fp, tn, fn
    for i, seg in enumerate(segments):
        seg2d = seg[:4].astype(int)
        seg_pts = np.array(list(bresenham(seg2d[0], seg2d[1], seg2d[2], seg2d[3])))
        seg_pts = seg_pts[(seg_pts[:, 0] >= 0) * (seg_pts[:, 1] >= 0) * (seg_pts[:, 0] < gts.shape[1]) * (seg_pts[:, 1] < gts.shape[0])]
        v = np.median(preds[seg_pts[:,1], seg_pts[:,0]])
        gtv = np.median(gts[seg_pts[:,1], seg_pts[:,0]])
        if v < np.cos(np.pi / 4):
            if gtv < np.cos(np.pi / 4):
                tp += 1
            else:
                fp += 1
        else:
            if gtv < np.cos(np.pi / 4):
                fn += 1
            else:
                tn += 1

if __name__ == '__main__':
    df = pd.read_csv('metadata_images_split_scene_v1.csv')
    df = df[df['scene_name'] < 'ai_004_001']
    rows = []
    for i, row in tqdm(df.iterrows()):
        if not row.included_in_public_release or row.split_partition_name != 'test':
            continue
        rows.append(row)
    
    for row in tqdm(rows):
        process(row.split_partition_name, row.scene_name, row.camera_name, row.frame_id)
    
    print(tp, fp, tn, fn)
    print("precision:", tp / (tp + fp))
    print("recall:", tp / (tp + fn))
    print("accuracy:", (tp + tn) / (tp + fp + tn + fn))
