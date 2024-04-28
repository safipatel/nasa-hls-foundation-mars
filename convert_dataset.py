# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import shutil
from functools import partial

import mmcv
import numpy as np
from PIL import Image, ImageDraw
import json



def parse_args():
    parser = argparse.ArgumentParser(description='Convert COCO-segmentation dataset format to mmsegmentation format')
    parser.add_argument('path', help='cityscapes data path')
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument('--outline',  help='outline')

    args = parser.parse_args()
    return args



def copy_and_create_mask(image_object, dataset_path, out_img_dir,
                       out_mask_dir, split, outline_only = False):
    imgpath, annotations = image_object

    if Image.open(osp.join(dataset_path, split, imgpath)).size != (512,512):
        print(imgpath, " not 512,512!")
    shutil.copyfile(osp.join(dataset_path, split, imgpath), osp.join(out_img_dir, split, imgpath))

    seg_filename = osp.join(out_mask_dir, split,imgpath.split(".jpg")[0] + '_gtmask.png')
    labelImg = Image.new("P", (512,512), 0)
    labelImg.putpalette([0,0,0,255,255,255])
    drawer = ImageDraw.Draw( labelImg )
    for ann in annotations:
        if outline_only:
            drawer.polygon(ann, outline=1 )
        else:
            drawer.polygon(ann, fill=1)
    labelImg.save(seg_filename, 'PNG')

def json2ann_imageslist(json_path):
    with open(json_path, 'r') as f:
        loaded_json = json.loads(f.read())
        images = loaded_json["images"]
        annotations = loaded_json["annotations"]
        result_dict = {}
        for ann in annotations:
            corr_img = ann["image_id"]
            file_name = images[corr_img]["file_name"]
            if file_name not in result_dict:
                result_dict[file_name] = []
            result_dict[file_name].append(ann["segmentation"][0])
    return list(result_dict.items())


def main():
    args = parse_args()
    dataset_path = osp.join(osp.dirname(osp.realpath(__file__)),args.path)
    out_dir = osp.join(osp.dirname(osp.realpath(__file__)),args.out_dir) if args.out_dir else dataset_path
    mmcv.mkdir_or_exist(out_dir)

    use_outline_only = args.outline == "True"
    out_img_dir = osp.join(out_dir, 'images')
    out_mask_dir = osp.join(out_dir, 'annotations')
    

    mmcv.mkdir_or_exist(osp.join(out_img_dir, 'train'))
    mmcv.mkdir_or_exist(osp.join(out_img_dir, 'test'))
    mmcv.mkdir_or_exist(osp.join(out_img_dir, 'valid'))
    mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'train'))
    mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'test'))
    mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'valid'))

    train_img_list = json2ann_imageslist(osp.join(dataset_path, 'train','_annotations.coco.json'))
    test_img_list = json2ann_imageslist(osp.join(dataset_path, 'test','_annotations.coco.json'))
    valid_img_list = json2ann_imageslist(osp.join(dataset_path, 'valid','_annotations.coco.json'))

    mmcv.track_progress(
        partial(
            copy_and_create_mask,
            dataset_path = dataset_path,
            out_img_dir=out_img_dir,
            out_mask_dir=out_mask_dir,
            split = "train", outline_only = use_outline_only), train_img_list)
    mmcv.track_progress(
        partial(
            copy_and_create_mask,
            dataset_path = dataset_path,
            out_img_dir=out_img_dir,
            out_mask_dir=out_mask_dir,
            split = "test", outline_only = use_outline_only), test_img_list)
    mmcv.track_progress(
        partial(
            copy_and_create_mask,
            dataset_path = dataset_path,
            out_img_dir=out_img_dir,
            out_mask_dir=out_mask_dir,
            split = "valid",outline_only = use_outline_only), valid_img_list)
    print('Done!')


    with open(osp.join(out_dir, 'train.txt'), 'w') as split_file:
        split_file.writelines(f[0] + '\n' for f in train_img_list)
    with open(osp.join(out_dir, 'test.txt'), 'w') as split_file:
        split_file.writelines(f[0] + '\n' for f in test_img_list)
    with open(osp.join(out_dir, 'valid.txt'), 'w') as split_file:
        split_file.writelines(f[0] + '\n' for f in valid_img_list)


if __name__ == '__main__':
    main()
