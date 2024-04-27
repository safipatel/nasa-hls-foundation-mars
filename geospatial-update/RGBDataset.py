
import os.path as osp

import numpy as np
import rioxarray
import torchvision.transforms.functional as F
from mmcv.parallel import DataContainer as DC
from mmseg.datasets.builder import PIPELINES
from torchvision import transforms
from PIL import Image
from torch import concat, Size

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

from mmseg.datasets.pipelines.loading import LoadAnnotations

from mmseg.datasets.pipelines.loading import LoadImageFromFile


@DATASETS.register_module()
class RGBDataset(CustomDataset):
    """RGBDataset dataset.
    """

    def __init__(self, CLASSES=(0, 1), PALETTE=None, **kwargs):
        
        self.CLASSES = CLASSES

        self.PALETTE = PALETTE
        
        gt_seg_map_loader_cfg = kwargs.pop('gt_seg_map_loader_cfg') if 'gt_seg_map_loader_cfg' in kwargs else dict()
        reduce_zero_label = kwargs.pop('reduce_zero_label') if 'reduce_zero_label' in kwargs else False
        
        super(RGBDataset, self).__init__(
            reduce_zero_label=reduce_zero_label,
            # ignore_index=2,
            **kwargs)

        self.gt_seg_map_loader = LoadAnnotations(reduce_zero_label=reduce_zero_label, **gt_seg_map_loader_cfg)





@PIPELINES.register_module()
class LoadRBGImage(LoadImageFromFile):
    """

    It loads a jpg image. Returns in channels last format.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        nodata (float/int): no data value to substitute to nodata_replace
        nodata_replace (float/int): value to use to replace no data
    """

    def __init__(self,  means, stds, to_float32=False, nodata=None, nodata_replace=0.0):
        self.to_float32 = to_float32
        self.nodata = nodata
        self.nodata_replace = nodata_replace
        self.conversion_means = means
        self.conversion_stds = stds

    def __call__(self, results):
        if results.get("img_prefix") is not None:
            filename = osp.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]
        
        image_file = Image.open(filename)
        image_file.load()
        img = np.asarray( image_file, dtype="uint16" )
        img = img.transpose(2,0,1)[[2,1,0]] # Put channels first and convert to BGR from RGB

        img[0,:]  = img[0,:] * ((self.conversion_means[0] + self.conversion_stds[0] * 2) / 255)
        img[1,:]  = img[1,:] * ((self.conversion_means[1] + self.conversion_stds[1] * 2) / 255)
        img[2,:]  = img[2,:] * ((self.conversion_means[2] + self.conversion_stds[2] * 2) / 255)

        
        # to channels last format
        img = np.transpose(img, (1, 2, 0))

        if self.to_float32:
            img = img.astype(np.float32)

        if self.nodata is not None:
            img = np.where(img == self.nodata, self.nodata_replace, img)

        results["filename"] = filename
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        results["flip"] = False
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}"
        return repr_str

# @PIPELINES.register_module()
# class LoadAnnotations(object):
#     """Load annotations for semantic segmentation.

#     Args:
#         to_uint8 (bool): Whether to convert the loaded label to a uint8
#         reduce_zero_label (bool): Whether reduce all label value by 1.
#             Usually used for datasets where 0 is background label.
#             Default: False.
#         nodata (float/int): no data value to substitute to nodata_replace
#         nodata_replace (float/int): value to use to replace no data


#     """

#     def __init__(
#         self,
#         reduce_zero_label=False,
#         nodata=None,
#         nodata_replace=-1,
#     ):
#         self.reduce_zero_label = reduce_zero_label
#         self.nodata = nodata
#         self.nodata_replace = nodata_replace

#     def __call__(self, results):
#         if results.get("seg_prefix", None) is not None:
#             filename = osp.join(results["seg_prefix"], results["ann_info"]["seg_map"])
#         else:
#             filename = results["ann_info"]["seg_map"]

#         gt_semantic_seg = open_tiff(filename).squeeze()

#         if self.nodata is not None:
#             gt_semantic_seg = np.where(
#                 gt_semantic_seg == self.nodata, self.nodata_replace, gt_semantic_seg
#             )
#         # reduce zero_label
#         if self.reduce_zero_label:
#             # avoid using underflow conversion
#             gt_semantic_seg[gt_semantic_seg == 0] = 255
#             gt_semantic_seg = gt_semantic_seg - 1
#             gt_semantic_seg[gt_semantic_seg == 254] = 255
#         if results.get("label_map", None) is not None:
#             # Add deep copy to solve bug of repeatedly
#             # replace `gt_semantic_seg`, which is reported in
#             # https://github.com/open-mmlab/mmsegmentation/pull/1445/
#             gt_semantic_seg_copy = gt_semantic_seg.copy()
#             for old_id, new_id in results["label_map"].items():
#                 gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id

#         results["gt_semantic_seg"] = gt_semantic_seg
#         results["seg_fields"].append("gt_semantic_seg")
#         return results


@PIPELINES.register_module()
class TorchNormalizeAndDuplicate(object):
    """Normalize the image.

    It normalises a multichannel image using torch

    Args:
        mean (sequence): Mean values .
        std (sequence): Std values of 3 channels.
    """

    def __init__(self, means, stds, duplicate = False):
        self.means = means
        self.stds = stds
        self.duplicate = duplicate

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        if self.duplicate:
            results["img"] = F.normalize(results["img"], self.means[0:3], self.stds[0:3], False)

            results["img_shape"] = results["img"].shape
            new_pad_shape = list(results["pad_shape"])
            new_pad_shape[0] = results["img_shape"][0]
            results["pad_shape"] = Size(new_pad_shape)
            results["img"] = concat((results["img"],results["img"]))
        else:
            results["img"] = F.normalize(results["img"], self.means, self.stds, False)

        results["img_norm_cfg"] = dict(mean=self.means, std=self.stds, duplicate = self.duplicate)
        return results
    
