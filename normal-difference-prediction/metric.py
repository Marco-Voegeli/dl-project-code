import json
import os
import numpy as np
import pickle
from tqdm import tqdm
from bresenham import bresenham

def metric(thresholds = [np.cos(np.pi / 12), np.cos(np.pi / 6), np.cos(np.pi / 4)],
            root_dataset = "./data/",
            filenames = "datalist/testing_hypersim_short.odgt",
            shape = (768, 1024),
            persample = False):
    list_sample = [json.loads(x.rstrip()) for x in open(filenames, 'r')]
    
    if not persample:
        tp = np.zeros(len(thresholds))
        fp = np.zeros(len(thresholds))
        tn = np.zeros(len(thresholds))
        fn = np.zeros(len(thresholds))
    else:
        precision = np.zeros(len(thresholds))
        recall_tpr = np.zeros(len(thresholds))
        fpr = np.zeros(len(thresholds))
        acc = np.zeros(len(thresholds))

    for sample in tqdm(list_sample):
        paths = sample['fpath_img'].split('/')
        img_name = paths[-1]
        
        paths[2] = "pred_normals_diff"
        camnum = paths[3].split('_')[2]
        paths[3] = f"cam_{camnum}"
        paths[-1] = img_name + '.npz'


        normal_diff = np.load(os.path.join(root_dataset, *paths))['normals_diff']


        paths[2] = "normals_diff"
        paths[-1] = '.'.join(img_name.split('.')[:2] + ['npz'])

        gt_diff = np.load(os.path.join(root_dataset, *paths))['normals_diff']

        paths[2] = 'lines'
        paths[3] = os.path.join(paths[3], 'segments')
        paths[-1] = '.'.join(img_name.split('.')[:2] + ['segments', 'pk'])

        segments = None
        with open(os.path.join(root_dataset, *paths), 'rb') as file:
            segments = pickle.load(file)['segments']
        
        if segments is None:
            return {}
        
        if persample:
            tp = np.zeros(len(thresholds))
            fp = np.zeros(len(thresholds))
            tn = np.zeros(len(thresholds))
            fn = np.zeros(len(thresholds))

        for i, seg in enumerate(segments):
            seg2d = seg[:4].astype(int)
            seg_pts = np.array(list(bresenham(seg2d[0], seg2d[1], seg2d[2], seg2d[3])))
            seg_pts = seg_pts[(seg_pts[:, 0] >= 0) * (seg_pts[:, 1] >= 0) * (seg_pts[:, 0] < shape[1]) * (seg_pts[:, 1] < shape[0])]
            v = np.median(normal_diff[seg_pts[:,1], seg_pts[:,0]])
            gtv = np.median(gt_diff[seg_pts[:,1], seg_pts[:,0]])

            for i, t in enumerate(thresholds):
                tp[i] += v < t and gtv < t
                fp[i] += v < t and gtv >= t
                tn[i] += v >= t and gtv >= t
                fn[i] += v >= t and gtv < t
        
        if persample:
            precision += tp / (tp + fp + 1e-30)
            recall_tpr += tp / (tp + fn + 1e-30)
            fpr += fp / (fp + tn + 1e-30)
            acc += (tp + tn) / (tp + tn + fp + fn + 1e-30)
    
    if persample:
        precision /= len(list_sample)
        recall_tpr /= len(list_sample)
        fpr /= len(list_sample)
        acc /= len(list_sample)
    else:
        precision = tp / (tp + fp + 1e-30)
        recall_tpr = tp / (tp + fn + 1e-30)
        fpr = fp / (fp + tn + 1e-30)
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-30)

    return {
        'precision': precision,
        'recall': recall_tpr,
        'fprate': fpr,
        'accuracy': acc
    }
