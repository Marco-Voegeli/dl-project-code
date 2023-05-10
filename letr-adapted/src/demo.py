from util.misc import nested_tensor_from_tensor_list
from models import build_model
import torch.nn.functional as F
import torchvision.transforms.functional as functional
import numpy as np
import os
import torch
import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

MODELS_DIR = '../exp/checkpoints/'
DEMOS_DIR = '../figures/demos/'

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return image


class ToTensor(object):
    def __call__(self, img):
        return functional.to_tensor(img)


def resize(image, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = functional.resize(image, size)

    return rescaled_image


class Resize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img):
        size = self.sizes
        return resize(img, size, self.max_size)

def plot_line_on_img(lines, ax, color='darkorange'):
    for line in lines:
        y1, x1, y2, x2 = line  # this is yxyx
        p1 = (x1.detach().numpy(), y1.detach().numpy())
        p2 = (x2.detach().numpy(), y2.detach().numpy())
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 linewidth=1, color=color, zorder=1)

def infer_on_image(model_name, raw_img, ax, demo):
    model_path = MODELS_DIR + model_name
    model_name_without_pth = model_name.split('.')[0]
    title = ' '.join(model_name_without_pth.split('_')[1:])
    
    # obtain checkpoints
    print('Loading model from {}'.format(model_path), flush=True)
    checkpoint = torch.load(model_path, map_location='cpu')

    # load model
    args = checkpoint['args']
    args.device = 'cpu'
    epochs = checkpoint['epoch']
    print('Model trained for {} epochs'.format(epochs), flush=True)
    model, _, postprocessors = build_model(args)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    h, w = raw_img.shape[0], raw_img.shape[1]
    orig_size = torch.as_tensor([int(h), int(w)])

    # normalize image
    test_size = 1100
    normalize = Compose([
        ToTensor(),
        Normalize([0.538, 0.494, 0.453], [0.257, 0.263, 0.273]),
        Resize([test_size]),
    ])
    img = normalize(raw_img)
    inputs = nested_tensor_from_tensor_list([img])

    if ' s1' in title:
        outputs = model(inputs)
    else:
        outputs = model(inputs)[0]

    out_logits, out_line = outputs['pred_logits'], outputs['pred_lines']
    prob = F.softmax(out_logits, -1)
    #scores, labels = prob[..., :-1].max(-1)
    scores, labels = prob[..., :].max(-1)
    img_h, img_w = orig_size.unbind(0)
    scale_fct = torch.unsqueeze(torch.stack(
        [img_w, img_h, img_w, img_h], dim=0), dim=0)
    lines = out_line * scale_fct[:, None, :]
    lines = lines.view(1000, 2, 2)
    lines = lines.flip([-1])  # this is yxyx format
    scores = scores.detach().numpy()
    keep = scores > 0.7
    keep = keep.squeeze()

    labels_text_ids = labels == 2
    labels_text_ids = labels_text_ids.squeeze()
    mask_text = labels_text_ids & keep
    lines_text = lines[mask_text]

    labels_struct_ids = labels == 0
    labels_struct_ids = labels_struct_ids.squeeze()
    struct_mask = labels_struct_ids & keep
    lines_struct = lines[struct_mask]

    lines_text = lines_text.reshape(lines_text.shape[0], -1)
    lines_struct = lines_struct.reshape(lines_struct.shape[0], -1)

    print('Number of text lines: {}'.format(lines_text.shape[0]), flush=True)
    print('Number of structural lines: {}'.format(lines_struct.shape[0]), flush=True)
    print('Plotting results', flush=True)

    img_h, img_w = raw_img.shape[0], raw_img.shape[1]
    fig_width = img_w / 20
    fig_height = img_h / 20

    fig, ax = plt.subplots(1, 1, frameon=False)
    ax.axis('off')
    # title = title + '\n ({} epochs)'.format(epochs)
    # ax.set_title(title)
    ax.imshow(raw_img)
    plot_line_on_img(lines_text, ax, color='blue')
    plot_line_on_img(lines_struct, ax, color='darkorange')
    
    demo_path = os.path.join('demo/', demo)
    os.makedirs(demo_path, exist_ok=True)
    fig.savefig(os.path.join(demo_path, model_name_without_pth + '.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def main():
    # (deepl) scp atabin@euler.ethz.ch:/cluster/scratch/atabin/LETR_euler/exp/res50_stage1_415_24h/checkpoints/checkpoint.pth ./exp/checkpoints/checkpoint_res50_s1.pth
    demos = os.listdir(DEMOS_DIR)
    models = os.listdir(MODELS_DIR)
    
    for demo in demos:
        demo_path = os.path.join(DEMOS_DIR, demo)
        # load image
        raw_img = plt.imread(demo_path)
        

        img_h, img_w = raw_img.shape[0], raw_img.shape[1]
        img_h = img_h / 20
        img_w = img_w / 20
        ratio = img_h / img_w
        print('Image size: {}x{}'.format(img_w, img_h), flush=True)

        # compute figure size in function of image size
        fig_width = img_h*2 + 6
        fig_height = img_h*3 + 9

        # 2 columns, 4 rows
        # first column for resnet50, second for resnet101
        # first row for input image, second for stage1, third for stage2, fourth for stage3    
        fig, axes = plt.subplots(4, 2, figsize=(fig_width, fig_height))
        fig.suptitle('Demo')

        print('Running inference on image', flush=True)
        for model in models:
            if 'res50' in model:
                column = 0
            else:
                column = 1

            if '_s1' in model:
                row = 1
            elif '_s2' in model:
                row = 2
            else:
                row = 3

            print("Model : ", model)
            print("Row   : ", row)
            print("Column: ", column)

            ax = axes[row][column]
            # demo = frame.name.00.00.jpg
            demo_nm = demo.split('.')[:-1]
            demo_nm = '_'.join(demo_nm)

            infer_on_image(model, raw_img, ax, demo_nm)

        # plt.show()
        print('Saving results', flush=True)
        # plt.savefig('demo.jpg', bbox_inches='tight', pad_inches=0)

main()
