import os

import tensorflow as tf

tf.__version__ = '2.0'
# This is needed since the notebook is stored in the object_detection folder.
if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# from utils import label_map_util
#
# from utils import visualization_utils as vis_util

# What model to download.
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/priya/Documents/mask_rcnn/exported_graphs_rcnn_inception' + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'toy_label_map.pbtxt')

NUM_CLASSES = 1
