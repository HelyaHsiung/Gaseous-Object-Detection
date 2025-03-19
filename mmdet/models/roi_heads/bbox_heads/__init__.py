# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_head import BBoxHead
from .convfc_bbox_head import  (Shared2FCBBoxHead,ConvFCBBoxHead)

__all__ = [
    'BBoxHead','Shared2FCBBoxHead','ConvFCBBoxHead'
]
