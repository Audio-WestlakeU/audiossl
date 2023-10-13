# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import random
import colorsys
import requests
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import skimage.io
from skimage.measure import find_contours
import cv2
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import torchaudio

import utils
import vision_transformer as vits
import dino_transform


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base','ast'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--patch_h', default=64, type=int, help='Patch resolution of the model.')
    parser.add_argument('--patch_w', default=3, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--audio_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    load_args = torch.load(args.pretrained_weights, map_location="cpu")["args"]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    import audio_transformer
    model = audio_transformer.AST_small(use_cls=load_args.use_cls if "use_cls" in load_args else True,
                                        patch_h=load_args.patch_height,patch_w=load_args.patch_width,mask_ratio=0.0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    # open image
    if args.audio_path is None:
        # user has not specified any image - we use our own image
        print("please specify --audio_path")
        exit(1)
    elif os.path.isfile(args.audio_path):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            16000, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        norm = dino_transform.MinMax(min=-79.6482,max=50.6842)
        waveform, sr = torchaudio.load(args.audio_path)
        melspec = norm(melspec_t(waveform))
        img = melspec
        
    else:
        print(f"Provided image path {args.audio_path} is non valid.")
        sys.exit(1)


    # make the image divisible by the patch size
    if img.shape[2] > 400:
        img = img[:,:,:400]
    h, w = img.shape[1] - img.shape[1] % load_args.patch_height, img.shape[2] - img.shape[2] % load_args.patch_width
    img = img[:, :h, :w].unsqueeze(0)

    h_featmap = img.shape[-2] // load_args.patch_height
    w_featmap = img.shape[-1] // load_args.patch_width
    print(img.shape)

    attentions = model.get_last_selfattention(img.to(device))
    if "use_cls" in load_args and load_args.use_cls == False:
        attentions = torch.cat([torch.zeros_like(attentions[:,:,0:1,:]),attentions],dim=2)
        attentions = torch.cat([torch.zeros_like(attentions[:,:,:,0:1]),attentions],dim=3)
        
    print(attentions.shape)
    dino_transform.plot_spec(torch.sum(attentions[0,:,1:,1:],dim=0).cpu().numpy(),"att_headsum.png")
    for i in range(attentions.shape[1]):
        dino_transform.plot_spec(attentions[0,i,1:,1:].cpu().numpy(),"att_head{}.png".format(i))
        plt.imsave(fname="att_head{}_imsave.png".format(i), arr=attentions[0,i].cpu().numpy(), format='png')
        print(attentions[0,i,:4,:4])
        print(torch.sum(attentions[0,i],dim=1))
        print(torch.sum(attentions[0,i],dim=0))


    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    print(attentions.shape)

    if args.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, h_featmap, w_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=(load_args.patch_height,load_args.patch_width), mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, h_featmap, w_featmap)
    print(attentions.shape)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=(load_args.patch_height,load_args.patch_width), mode="nearest")[0].cpu().numpy()
    print(attentions.shape)

    # save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    dino_transform.plot_spec(img[0,0].cpu().numpy(),"img0.png")
    for j in range(nh):
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    if args.threshold is not None:
        image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)
