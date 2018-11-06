#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.inc_res_v2 import inc_res_v2
from nets.resnet_v1_dense import resnetv1_dense

CLASSES = ('__background__', # always index 0
           'abudefduf vaigiensis', 'acanthurus nigrofuscus', 'amphiprion clarkii', 'chaetodon lunulatus',
           'chaetodon speculum', 'chaetodon trifascialis', 'chromis chrysura', 'dascyllus aruanus', 'dascyllus reticulatus',
           'hemigymnus melapterus', 'myripristis kuntee', 'neoglyphidodon nigroris', 'pempheris vanicolensis',
           'plectrogly-phidodon dickii', 'zebrasoma scopas', 'other')
CONCERNED_ERRORS = ['__background__', # always index 0
           'abudefduf vaigiensis', 'acanthurus nigrofuscus', 'amphiprion clarkii', 'chaetodon lunulatus',
           'chaetodon speculum', 'chaetodon trifascialis', 'chromis chrysura', 'dascyllus aruanus', 'dascyllus reticulatus',
           'hemigymnus melapterus', 'myripristis kuntee', 'neoglyphidodon nigroris', 'pempheris vanicolensis',
           'plectrogly-phidodon dickii', 'zebrasoma scopas', 'other']
# CONCERNED_ERRORS = [] # Don't save any error cases

# NET = './output-152-receipts/res152/train/default/res152_faster_rcnn_iter_30000.ckpt'
NET = './output/res152_dense_clef_orig/train/default/res152_dense_faster_rcnn_iter_50001.ckpt'
DATASET = 'fishclef'
NETWORK = 'res152_dense'
IMAGE_EXTENSIONS = ['.jpg', '.png', '.bmp']
IoU_THRESHOLDS = ['0.5']

USE_ORIGINAL_IMAGES = True

def bbox_intersection_over_union(boxA, boxB):
  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  # compute the area of intersection rectangle
  interArea = (xB - xA + 1) * (yB - yA + 1)
 
  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)
 
  # return the intersection over union value
  return iou

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(image_name)

    if im is None:
        print ("Error: Unable to load file %s" % (image_name))
        exit(-1)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    finalBBoxes = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        # vis_detections(im, cls, dets, thresh=CONF_THRESH)
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            finalBBoxes.append([bbox[0], bbox[1], bbox[2], bbox[3], score, cls])

            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 5)
            w = bbox[2] - bbox[0]
            # cv2.putText(im, cls + "(" + str(score) + ")", (int(bbox[0] + (w / 2.0) - 100), int(bbox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(im, cls, (int(bbox[0]), int(bbox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    return im, finalBBoxes

def loadGTAnnotationsFromXML(xml_path):
    if not os.path.exists(xml_path):
        print ("Error: Unable to locate XML file %s" % (im_name))
        exit(-1)   
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    num_objs = len(objs)

    # Load object bounding boxes into a data frame.
    boundingBoxes = []
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        # cls = self._class_to_ind[obj.find('name').text.lower().strip()]
        cls = obj.find('name').text.lower().strip()
        if cls not in CLASSES:
            cls = 'other'
        boundingBoxes.append([x1, y1, x2, y2, 1.0, cls])

    return boundingBoxes

def convertToXML(im_name, bboxes):
    xml = '<document filename="' + im_name +'">'
    for bbox in bboxes:
        bbox[:4] = [int(coord) for coord in bbox[:4]]
        # Get document bounds
        topLeft = str(bbox[0]) + ',' + str(bbox[1])
        topRight = str(bbox[2]) + ',' + str(bbox[1])
        bottomLeft = str(bbox[0]) + ',' + str(bbox[3])
        bottomRight = str(bbox[2]) + ',' + str(bbox[3])

        className = bbox[5].replace(" ", "_") + "_Region"
        xml += '<' + className + ' prob="' + str(bbox[4]) + '">'
        xml += '<Coords points="' + topLeft + ' ' + topRight + ' ' + bottomLeft + ' ' + bottomRight + '"/>'
        xml += '</' + className + '>'

    xml += '</document>'
    dom = parseString(xml)
    xml = dom.toprettyxml()
    xml = xml.split('\n')
    xml = xml[1:]
    xml = '\n'.join(xml)

    return xml    

def computeStatistics(detections, gt, statistics, iou_thresholds):
    classificationErrorOccured = False
    for thresh in iou_thresholds:
        matchedGTBBox = [0] * len(gt)
        # Iterate over all the predicted bboxes
        for predictedBBox in detections:
            bboxMatchedIdx = -1
            # Iterate over all the GT bboxes
            for gtBBoxIdx, gtBBox in enumerate(gt):
                # Compute IoU
                iou = bbox_intersection_over_union(gtBBox, predictedBBox)
                if ((iou > float(thresh)) and (gtBBox[5] == predictedBBox[5])):
                    if not matchedGTBBox[gtBBoxIdx]:
                        bboxMatchedIdx = gtBBoxIdx
                        break

            if (bboxMatchedIdx != -1):
                statistics[predictedBBox[5]][thresh]["truePositives"] += 1
                matchedGTBBox[bboxMatchedIdx] = 1
            else:
                statistics[predictedBBox[5]][thresh]["falsePositives"] += 1
                if predictedBBox[5] in CONCERNED_ERRORS:
                    classificationErrorOccured = True

        # All the unmatched bboxes are false negatives
        # falseNegatives = len(matchedGTBBox) - sum(matchedGTBBox)
        for idx, gtBBox in enumerate(gt):
            if not matchedGTBBox[idx]:
                statistics[gtBBox[5]][thresh]["falseNegatives"] += 1
                if gtBBox[5] in CONCERNED_ERRORS:
                    classificationErrorOccured = True

    return statistics, classificationErrorOccured

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    # parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
    #                     choices=NETS.keys(), default='res152')
    # parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
    #                     choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    demonet = NETWORK
    dataset = DATASET
    tfmodel = NET

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nPlease specify the correct directory.').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    elif demonet == 'res152':
        net = resnetv1(batch_size=1, num_layers=152)
    elif demonet == 'inc_res_v2':
        net = inc_res_v2(batch_size=1)
    elif demonet == 'res152_dense':
        net = resnetv1_dense(batch_size=1, num_layers=152)

    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", len(CLASSES),
                          tag='default', anchor_scales=[4,8,16,32,64], anchor_ratios=[0.25,0.5,1,2,4])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    outputFile = open('/netscratch/siddiqui/Datasets/FishCLEF/faster-rcnn/output-dense-50k.txt', 'w')
    outputFile.write('<?xml version="1.0" encoding="UTF-8"?>\n')

    statistics = {}
    for cls_ind, cls in enumerate(CLASSES[1:]):
        statistics[cls] = {}
        for thresh in IoU_THRESHOLDS:
            statistics[cls][thresh] = {}
            statistics[cls][thresh]["truePositives"] = 0
            statistics[cls][thresh]["falsePositives"] = 0
            statistics[cls][thresh]["falseNegatives"] = 0
            statistics[cls][thresh]["precision"] = 0
            statistics[cls][thresh]["recall"] = 0
            statistics[cls][thresh]["fMeasure"] = 0
    
    im_names_file = open('/netscratch/siddiqui/Datasets/FishCLEF/data/ImageSets/test.txt', 'r')

    for im_name in im_names_file:
        im_name = im_name.strip()
        print ("Processing file: %s" % (im_name))

        found = False
        for ext in IMAGE_EXTENSIONS:
            if USE_ORIGINAL_IMAGES:
                im_name_with_ext = im_name + '-orig' + ext
            else:
                im_name_with_ext = im_name + ext
            im_path = os.path.join('/netscratch/siddiqui/Datasets/FishCLEF/data/Images', im_name_with_ext)
            if os.path.exists(im_path):
                found = True
                break
        if not found:
            print ("Error: Unable to locate file %s" % (im_name))
            exit(-1)

        xml_path = os.path.join('/netscratch/siddiqui/Datasets/FishCLEF/data/Annotations', im_name + '.xml')
        gtBBoxes = loadGTAnnotationsFromXML(xml_path)
        
        im, bboxes = demo(sess, net, im_path)

        # Compute the statistics
        statistics, classificationErrorOccured = computeStatistics(bboxes, gtBBoxes, statistics, IoU_THRESHOLDS)
        if classificationErrorOccured:
            print ("Writing incorrect image: %s" % (im_name))
            for bbox in gtBBoxes:
                cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 3)
                w = bbox[2] - bbox[0]
                cv2.putText(im, bbox[5], (int(bbox[0] + (w / 2.0) - 100), int(bbox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            cv2.imwrite(os.path.join('/netscratch/siddiqui/Datasets/FishCLEF/faster-rcnn/IncorrectDetection', im_name + '.jpg'), im)            

        # Write the output in ICDAR Format
        outputFile.write(convertToXML(im_name_with_ext, bboxes))
        # cv2.imwrite(os.path.join('/netscratch/siddiqui/Datasets/FishCLEF/faster-rcnn/Output-Images-Dense-50k', im_name + '.jpg'), im)

    outputFile.close()

    # Compute final precision and recall
    outputFile = open('/netscratch/siddiqui/Datasets/FishCLEF/faster-rcnn/output-dense-50k-stats.txt', 'w')
    for cls in statistics.keys():
        for thresh in statistics[cls].keys():
            if (statistics[cls][thresh]["truePositives"] == 0) and (statistics[cls][thresh]["falsePositives"] == 0):
                precision = 1.0
            else:
                precision = float(statistics[cls][thresh]["truePositives"]) / float(statistics[cls][thresh]["truePositives"] + statistics[cls][thresh]["falsePositives"])
            if (statistics[cls][thresh]["truePositives"] == 0) and (statistics[cls][thresh]["falseNegatives"] == 0):
                recall = 1.0
            else:
                recall = float(statistics[cls][thresh]["truePositives"]) / float(statistics[cls][thresh]["truePositives"] + statistics[cls][thresh]["falseNegatives"])
            if (precision == 0.0) and (recall == 0.0):
                fMeasure = 0.0
            else:
                fMeasure = 2 * ((precision * recall) / (precision + recall))

            statistics[cls][thresh]["precision"] = precision
            statistics[cls][thresh]["recall"] = recall
            statistics[cls][thresh]["fMeasure"] = fMeasure

            print ("--------------------------------")
            print ("Class: %s" % (cls))
            print ("IoU Threshold: %s" % (thresh))
            print ("True Positives: %d" % (statistics[cls][thresh]["truePositives"]))
            print ("False Positives: %d" % (statistics[cls][thresh]["falsePositives"]))
            print ("False Negatives: %d" % (statistics[cls][thresh]["falseNegatives"]))
            print ("Precision: %f" % (precision))
            print ("Recall: %f" % (recall))
            print ("F-Measure: %f" % (fMeasure))

            outputFile.write ("Class: %s" % (cls) + "\n")
            outputFile.write ("IoU Threshold: %s" % (thresh) + "\n")
            outputFile.write ("True Positives: %d" % (statistics[cls][thresh]["truePositives"]) + "\n")
            outputFile.write ("False Positives: %d" % (statistics[cls][thresh]["falsePositives"]) + "\n")
            outputFile.write ("False Negatives: %d" % (statistics[cls][thresh]["falseNegatives"]) + "\n")
            outputFile.write ("Precision: %f" % (precision) + "\n")
            outputFile.write ("Recall: %f" % (recall) + "\n")
            outputFile.write ("F-Measure: %f" % (fMeasure) + "\n")
            outputFile.write ("--------------------------------\n")

    outputFile.close()