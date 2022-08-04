# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in MMDetection3D."""
from typing import List, Optional, Union

from mmengine.config import ConfigDict
from mmengine.data import InstanceData

from mmdet.models.task_modules import SamplingResult

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]

# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

InstanceList = List[InstanceData]
OptInstanceList = Optional[InstanceList]

SamplingResultList = List[SamplingResult]

OptSamplingResultList = Optional[SamplingResultList]