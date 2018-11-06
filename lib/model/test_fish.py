# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math
import os.path

import xmltodict

from utils.timer import Timer
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

SAVE_SINGLE_IMAGE = True
import matplotlib.pyplot as plt
# CLASSES = ('__background__', # always index 0
#            'header', 'footer', 'logo', 'total_amount', 'total_amount_text', 'row', 'name', 'price')
CLASSES = ('__background__', # always index 0
           'fish')
# Colors are in BGR format
# CLASSES_COLORS = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}
CLASSES_COLORS = {1: (255, 255, 255), 2: (0, 0, 0)}
COMPARE_WITH_ANNOTATIONS = True
IOU_THRESHOLD = 0.5
IOM_THRESHOLD = 0.5
USE_IOU = True

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

def bbox_intersection_over_min(boxA, boxB):
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
 
  # compute the intersection over min by taking the intersection
  # area and dividing it by the min of prediction and ground-truth
  # areas
  iom = interArea / float(min(boxAArea, boxBArea))
 
  # return the intersection over union value
  return iom

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  # seems to have height, width, and image scales
  # still not sure about the scale, maybe full image it is 1.
  blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])

  boxes = rois[:, 1:5] / im_scales[0]
  # print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def vis_detections(im, class_name, dets, thresh=0.5):
  """Draw detected bounding boxes."""
  inds = np.where(dets[:, -1] >= thresh)[0]
  if len(inds) == 0:
      return

  im = im[:, :, (2, 1, 0)]
  fig, ax = plt.subplots(figsize=(12, 12))
  ax.imshow(im, aspect='equal')
  for i in inds:
      bbox = dets[i, :4]
      score = dets[i, -1]

      ax.add_patch(
          plt.Rectangle((bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1], fill=False,
                        edgecolor='red', linewidth=3.5)
          )
      ax.text(bbox[0], bbox[1] - 2,
              '{:s} {:.3f}'.format(class_name, score),
              bbox=dict(facecolor='blue', alpha=0.5),
              fontsize=14, color='white')

  ax.set_title(('{} detections with '
                'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                thresh),
                fontsize=14)
  plt.axis('off')
  plt.tight_layout()
  plt.draw()

def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.05):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  # num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)

  # all_boxes = [[[] for _ in range(num_images)]
  #        for _ in range(imdb.num_classes)]
  #
  # output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  # testFile = open('/netscratch/siddiqui/Datasets/ComplexBackground/data/ImageSets/test.txt')
  testFile = open('/netscratch/siddiqui/Datasets/ComplexBackground/data_of_bgs_gray/data/ImageSets/test.txt')
  imageNames = testFile.readlines()
  counter = 0
  reject_classes = []

  imagesOutputDir = '/netscratch/siddiqui/Datasets/ComplexBackground/faster-rcnn/output-images/'
  os.system("rm -rf " + imagesOutputDir)
  os.system("mkdir " + imagesOutputDir)

  fileAlreadyProcessed = False
  if os.path.isfile("/netscratch/siddiqui/Datasets/ComplexBackground/faster-rcnn/output.txt"):
    f = open("/netscratch/siddiqui/Datasets/ComplexBackground/faster-rcnn/output.txt", "r")
    processedFiles = f.readlines()
    f.close()
    #print (processedFiles)
    if len(processedFiles) != 0:
      print ("Resuming processing")
      lastProcessedFile = processedFiles[-1]
      lastProcessedFile = lastProcessedFile.split(';')[0]
      fileAlreadyProcessed = True
      print ("Last processed file: %s" % lastProcessedFile)
  
  fileIndex = 0
  videoScores = {}
  scoreFile = open("/netscratch/siddiqui/Datasets/ComplexBackground/faster-rcnn/output-image.txt", "w")
  for im_name in imageNames:
    im_name = im_name.strip()
    # Skip all names already processed
    if fileAlreadyProcessed:
      fileIndex += 1
      if im_name == lastProcessedFile:
        print("Resuming processing from file (%d): %s" % (fileIndex, im_name))
        fileAlreadyProcessed = False
      continue

    rejectExample = False
    for r_class in reject_classes:
      if r_class in im_name:
        rejectExample = True
        break
    if rejectExample:
      continue
    # im_path = '/netscratch/siddiqui/Datasets/ComplexBackground/data/Images/' + im_name + '.png'
    # annot_file = '/netscratch/siddiqui/Datasets/ComplexBackground/data/Annotations/' + im_name + '.xml'
    im_path = '/netscratch/siddiqui/Datasets/ComplexBackground/data_of_bgs_gray/data/Images/' + im_name + '.png'
    annot_file = '/netscratch/siddiqui/Datasets/ComplexBackground/data_of_bgs_gray/data/Annotations/' + im_name + '.xml'
    
    video_name = im_name[:im_name.rfind('_')]
    if video_name not in videoScores:
      # True Positives, False Positives, False Negatives
      videoScores[video_name] = [0, 0, 0]

    im = cv2.imread(im_path)
    if im is None:
      print ("Error loading file: %s" % im_path)
      continue

    overlay = im.copy()

    _t['im_detect'].tic()
    scores, boxes = im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3

    with open(annot_file, 'r') as fd:
      doc = xmltodict.parse(fd.read())

    # Load GT bboxes
    gtBBoxes = []
    for xmlAttribName, xmlData in doc['annotation'].items():
      # print (xmlAttribName)
      if isinstance(xmlData, list):
        for obj in xmlData:
          # If multiple objects
          bbox = obj['bndbox']
          gtBBoxes.append([int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])
      else:
        # If only one object
        bbox = xmlData['bndbox']
        gtBBoxes.append([int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])

    bboxes = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls = CLASSES[cls_ind]
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        if SAVE_SINGLE_IMAGE:
          inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
          for i in inds:
              bbox = dets[i, :4]
              score = dets[i, -1]
              bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], score, cls])
              cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), CLASSES_COLORS[cls_ind], 3) # Negative thinkness results in filled rect
        else:
          vis_detections(im, cls, dets, thresh=CONF_THRESH)

    for gtBBox in gtBBoxes:
      cv2.rectangle(overlay, (gtBBox[0], gtBBox[1]), (gtBBox[2], gtBBox[3]), CLASSES_COLORS[2], 3) # Negative thinkness results in filled rect

    if SAVE_SINGLE_IMAGE:
      if True: #len(bboxes) > 0:
        # (3) blend with the original:
        opacity = 0.5
        cv2.addWeighted(overlay, opacity, im, 1 - opacity, 0, im)
        # out_im_path = '/netscratch/siddiqui/Bosch/data/faster-rcnn/output-defected-io/' + img_name + '.jpg'
        # out_im_path = '/netscratch/siddiqui/TableDetection/output-images/' + im_name.split('/')[-1]
        out_im_path = imagesOutputDir + im_name + '.png'
        cv2.imwrite(out_im_path, im)
        print ("Writing output image for file (%d): %s" % (fileIndex, im_name))

        f = open("/netscratch/siddiqui/Datasets/ComplexBackground/faster-rcnn/output.txt", "a+")
        f.write(im_name + ';' + str(len(bboxes)) + ';' + str(bboxes) + "\n")
        f.close()
    else:
      # Close previous plots before moving onto the next image
      plt.show()
      plt.close('all')

    # Compute F measure based on bounding boxes
    truePositives = 0
    falsePositives = 0
    falseNegatives = 0
    matchedGTBBox = [0] * len(gtBBoxes)

    # Iterate over all the predicted bboxes
    for predictedBBox in bboxes:
      bboxMatchedIdx = -1
      # Iterate over all the GT bboxes
      for gtBBoxIdx in range(len(gtBBoxes)):
        gtBBox = gtBBoxes[gtBBoxIdx]
        if USE_IOU:
          # Compute IoU
          iou = bbox_intersection_over_union(gtBBox, predictedBBox)
          if (iou > IOU_THRESHOLD):
            if (matchedGTBBox[gtBBoxIdx] == 0):
              bboxMatchedIdx = gtBBoxIdx
              break
        else:
          # Compute IoM
          iom = bbox_intersection_over_min(gtBBox, predictedBBox)
          # if ((iom > IOM_THRESHOLD) and (not matchedGTBBox[bboxMatchedIdx])):
          #   bboxMatchedIdx = gtBBoxIdx
          #   break
          if (iom > IOM_THRESHOLD):
            if (matchedGTBBox[gtBBoxIdx] == 0):
              bboxMatchedIdx = gtBBoxIdx
              break
          
      if (bboxMatchedIdx != -1):
        truePositives += 1
        matchedGTBBox[bboxMatchedIdx] = 1
      else:
        falsePositives += 1

    # All the unmatched bboxes are false negatives
    falseNegatives = len(matchedGTBBox) - sum(matchedGTBBox)

    # Print final statistics for the frame
    print ("True positives: %d" % truePositives)
    print ("False positives: %d" % falsePositives)
    print ("False negatives: %d" % falseNegatives)

    videoScores[video_name][0] += truePositives
    videoScores[video_name][1] += falsePositives
    videoScores[video_name][2] += falseNegatives

    # Compute F-Score
    if ((truePositives == 0) and (falseNegatives == 0) and (falsePositives == 0)):
      assert((len(gtBBoxes) == 0) and (len(bboxes) == 0))
      recall = 100.0
      precision = 100.0
    else:
      if ((truePositives == 0) and (falseNegatives == 0)):
        recall = 0.0
      else:
        recall = (truePositives / float(truePositives + falseNegatives)) * 100

      if ((truePositives == 0) and (falsePositives == 0)):
        precision = 0.0
      else:
        precision = (truePositives / float(truePositives + falsePositives)) * 100

    if ((precision == 0.0) and (recall == 0.0)):
      fMeasure = 0.0
    else:
      fMeasure = 2 * ((precision * recall) / (precision + recall))

    print ("Recall: %f" % recall)
    print ("Precision: %f" % precision)
    print ("F-Measure: %f" % fMeasure)

    scoreFile.write(im_name + ';' + str([len(bboxes), len(gtBBoxes), truePositives, falsePositives, falseNegatives, recall, precision, fMeasure]) + '\n')

    fileIndex += 1

  print ("-------------------------------------------")
  # Write video scores to file
  videoScoresFileName = "/netscratch/siddiqui/Datasets/ComplexBackground/faster-rcnn/video.txt"
  averageFMeasure = 0
  videoScoresFile = open(videoScoresFileName, 'w')

  for videoName, videoScore in videoScores.items():
    print (videoName)
    recall = (videoScore[0] / float(videoScore[0] + videoScore[2])) * 100
    precision = (videoScore[0] / float(videoScore[0] + videoScore[1])) * 100
    fMeasure = 2 * ((precision * recall) / (precision + recall))
    videoScoresFile.write(videoName + ";" + str(videoScore + [recall, precision, fMeasure]) + '\n')
    print ("Recall: %f" % recall)
    print ("Precision: %f" % precision)
    print ("F-Measure: %f" % fMeasure)

    averageFMeasure += fMeasure

  print ("-------------------------------------------")
  averageFMeasure = averageFMeasure / len(videoScores)
  print ("Average F-Measure: %f" % averageFMeasure)

  videoScoresFile.write('Average F-Measure: ' + str(averageFMeasure) + '\n')
  videoScoresFile.close()

  scoreFile.close()
  #
  #   # skip j = 0, because it's the background class
  #   for j in range(1, imdb.num_classes):
  #     inds = np.where(scores[:, j] > thresh)[0]
  #     cls_scores = scores[inds, j]
  #     cls_boxes = boxes[inds, j*4:(j+1)*4]
  #     cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
  #       .astype(np.float32, copy=False)
  #     keep = nms(cls_dets, cfg.TEST.NMS)
  #     cls_dets = cls_dets[keep, :]
  #     all_boxes[j][i] = cls_dets
  #
  #   # Limit to max_per_image detections *over all classes*
  #   if max_per_image > 0:
  #     image_scores = np.hstack([all_boxes[j][i][:, -1]
  #                   for j in range(1, imdb.num_classes)])
  #     if len(image_scores) > max_per_image:
  #       image_thresh = np.sort(image_scores)[-max_per_image]
  #       for j in range(1, imdb.num_classes):
  #         keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
  #         all_boxes[j][i] = all_boxes[j][i][keep, :]
  #   _t['misc'].toc()
  #
  #   print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
  #       .format(i + 1, num_images, _t['im_detect'].average_time,
  #           _t['misc'].average_time))
  #
  # det_file = os.path.join(output_dir, 'detections.pkl')
  # with open(det_file, 'wb') as f:
  #   pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  # print('Evaluating detections')
  # imdb.evaluate_detections(all_boxes, output_dir)
