import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        self.random_flip = opt.random_flip
        # parse the input list
        self.parse_input_list(odgt, **kwargs)

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = Image.open(image_path).convert('RGB')
        segm = Image.fromarray(np.load(segm_path)['normals_diff'])
        assert(segm.mode == "F")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        flipped = self.random_flip and np.random.choice([0, 1])
        if flipped:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

        segm = np.array(segm)
        segm[segm > 1] = 1
        segm[segm < -1] = -1
        segm[np.isnan(segm)] = 1

        paths = this_record['fpath_img'].split('/')
        paths[2] = "pred_normals_diff"
        camnum = paths[3].split('_')[2]
        paths[3] = f"cam_{camnum}"

        output = dict()
        output['image'] = np.array(img).transpose((2, 0, 1)).astype('uint8')
        # segm = (segm < np.cos(np.pi / 4)).astype('float32')
        output['mask'] = np.expand_dims(segm, axis=0)# batch_segms.contiguous()        
        output['savedir'] = os.path.join(self.root_dataset, *paths)
        output['flipped'] = flipped
        return output

    def __len__(self):
        return self.num_sample

class TestDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, savedir, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        self.savedir = savedir

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = Image.open(image_path).convert('RGB')
        segm = Image.fromarray(np.load(segm_path)['normals_diff'])
        assert(segm.mode == "F")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        flipped = self.random_flip and np.random.choice([0, 1])
        if flipped:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

        segm = np.array(segm)
        segm[segm > 1] = 1
        segm[segm < -1] = -1
        segm[np.isnan(segm)] = 1

        paths = this_record['fpath_img'].split('/')
        paths[2] = self.savedir + "_pred_normals_diff"
        camnum = paths[3].split('_')[2]
        paths[3] = f"cam_{camnum}"

        output = dict()
        output['image'] = np.array(img).transpose((2, 0, 1)).astype('uint8')
        # segm = (segm < np.cos(np.pi / 4)).astype('float32')
        output['mask'] = np.expand_dims(segm, axis=0)# batch_segms.contiguous()        
        output['savedir'] = os.path.join(self.root_dataset, *paths)
        output['flipped'] = flipped
        return output

    def __len__(self):
        return self.num_sample