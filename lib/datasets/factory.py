# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
# from datasets.coco import coco

import numpy as np

# Set up voc_<year>_<split> 
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

from datasets.fish import fish # Custom
# fish_devkit_path = '/netscratch/siddiqui/Datasets/ComplexBackground'
# fish_devkit_path = '/netscratch/siddiqui/Datasets/ComplexBackground/data_of_bgs_gray/'
fish_devkit_path = '/netscratch/siddiqui/Datasets/FishCLEF/'
for split in ['train', 'test']:
    name = '{}_{}'.format('fish', split)
    __sets[name] = (lambda split=split: fish(split, fish_devkit_path))

from datasets.fishclef import fishclef # Custom
fishclef_devkit_path = '/netscratch/siddiqui/Datasets/FishCLEF'
for split in ['train', 'test']:
    name = '{}_{}'.format('fishclef', split)
    __sets[name] = (lambda split=split: fishclef(split, fishclef_devkit_path))

from datasets.tableReject import table # Custom
# table_devkit_path = '/netscratch/siddiqui/TableDetection/data_icdar_17'
# from datasets.table import table
table_devkit_path = '/netscratch/siddiqui/TableDetection'
for split in ['train', 'test']:
    name = '{}_{}'.format('table', split)
    __sets[name] = (lambda split=split: table(split, table_devkit_path))

from datasets.receipts import receipts # Custom
receipts_devkit_path = '/netscratch/siddiqui/DeepReceipts/' #'$PY_FASTER_RCNN/data/INRIA_Person_devkit'
for split in ['train', 'test']:
    name = '{}_{}'.format('receipts', split)
    __sets[name] = (lambda split=split: receipts(split, receipts_devkit_path))

from datasets.bosch import Bosch # Custom
# bosch_devkit_path = '/netscratch/siddiqui/Bosch/Cam3_RDS_Dataset/'
bosch_devkit_path = '/netscratch/siddiqui/Bosch/Data_2016_Dataset/'
for split in ['train', 'test']:
    name = '{}_{}'.format('bosch', split)
    __sets[name] = (lambda split=split: Bosch(split, bosch_devkit_path))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
