import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")
from utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util

from utils import visualization_utils as vis_util


import glob
import json
import argparse
import datetime

from typing import Dict, Optional

import cv2

from tqdm import tqdm, trange

from make_xml import make_xml



PATH_TO_FROZEN_GRAPH = 'models/frozen_inference_graph.pb'
PATH_TO_LABELS = 'models/person_label_map.pbtxt'
MIN_THRESHOLD = 0.8

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def create_session(graph, image_size):
    print('0')
    with graph.as_default():
        print('1')
        sess = tf.Session()
        print('2')
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image_size[0], image_size[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        return sess, tensor_dict, image_tensor

def run_inference_for_multi_image(images, graph):
    
    sess, tensor_dict, image_tensor = create_session(graph, images[0].shape[:2])
    
    for image in images:
        # Run inference
        output_dict = sess.run(tensor_dict,
                                feed_dict={image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        
        boxes = []
        for i in range(len(output_dict['detection_scores'])):
            if output_dict['detection_scores'][i] > MIN_THRESHOLD:
                boxes.append(output_dict['detection_boxes'][i])
        print(boxes)
        im = draw_boxes(image, boxes)[:, :, ::-1].copy()
        cv2.imshow('aa', im)
        cv2.waitKey(0)
        # Image.fromarray(im).show()


def run_inference_for_single_image(image, sess, tensor_dict, image_tensor):
    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    boxes = []
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > MIN_THRESHOLD:
            box = output_dict['detection_boxes'][i]
            top = box[0]
            left = box[1]
            height = box[2]
            width = box[3]
            left = int(left * image.shape[1])
            right = int(width * image.shape[1])
            top = int(top * image.shape[0])
            bottom = int(height * image.shape[0])
            boxes.append((top, bottom, left, right))
    # print(boxes)
    # im = draw_boxes(image, boxes)
    # print(im)
    # cv2.imshow('aa', im)
    # cv2.waitKey(0)
    # cv2.imwrite('{}.jpg'.format(images.index(image)), im)
    return boxes

def predic_image(image_path):
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     output_dict['detection_boxes'],
    #     output_dict['detection_classes'],
    #     output_dict['detection_scores'],
    #     category_index,
    #     instance_masks=output_dict.get('detection_masks'),
    #     use_normalized_coordinates=True,
    #     line_thickness=8)
    # print(output_dict)
    boxes = []
    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > MIN_THRESHOLD:
            boxes.append(output_dict['detection_boxes'][i])
    # print(len(boxes))
    # print(boxes)
    return boxes


def draw_boxes(image, boxes):
    for box in boxes:
        top = box[0]
        left = box[1]
        height = box[2]
        width = box[3]

        xmin = int(left * image.shape[1])
        xmax = int(width * image.shape[1])
        ymin = int(top * image.shape[0])
        ymax = int(height * image.shape[0])

        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            (0, 0, 255),
            1
        )

    return image

def detect_from_video(sess, tensor_dict, image_tensor, save_dir: str,
                      video_path: str,
                      detect_results: Dict[str, dict],
                      every_n_frames: Optional[int] = None) -> dict:
    cap = cv2.VideoCapture(video_path)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print(width, height)
    # sess, tensor_dict, image_tensor = create_session(graph, [width, height])

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    for count in trange(n_frames):
        ret, frame = cap.read()
        if every_n_frames is not None and count % every_n_frames != 0:
            continue
        im_GBR = frame[:, :, ::-1].copy()
        results = run_inference_for_single_image(im_GBR, sess, tensor_dict, image_tensor)

        if len(results) > 0:
            frame_id = "{:04d}".format(count)
            fname = "{}_{}.jpg".format(video_name, frame_id)
            cv2.imwrite(os.path.join(save_dir, fname), frame)
            # for n, (prob, bb) in enumerate(results):
            for n, (top, bottom, left, right) in enumerate(results):
                person_id = "{:02d}".format(n)
                if fname not in detect_results:
                    detect_results[fname] = dict()

                detect_results[fname].update({
                    person_id: {
                        "top": top,
                        "bottom": bottom,
                        "left": left,
                        "right": right,
                        # "prob": prob,
                    }
                })

    cap.release()
    return detect_results


def detect_all(base_dir: str,
               save_base_dir: str,
               every_n_frames: Optional[int] = None) -> None:

    print(base_dir, save_base_dir, every_n_frames)
    sess = None
    tensor_dict = None
    image_tensor = None

    for hour_dir in glob.glob(os.path.join(base_dir, "*")):
        print(hour_dir)
        hour_dir_name = os.path.split(hour_dir)[-1]
        save_dir = os.path.join(save_base_dir, hour_dir_name)
        images_dir = os.path.join(save_dir, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        json_path = os.path.join(save_dir, "result.json")
        if os.path.exists(json_path):
            continue

        detect_results: Dict[str, dict] = dict()
        files = glob.glob(os.path.join(hour_dir, "*.mp4"))
        for video_path in tqdm(files):
            if sess is None:
                cap = cv2.VideoCapture(video_path)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()
                print(width, height)
                sess, tensor_dict, image_tensor = create_session(detection_graph, [width, height])
            detect_from_video(sess, tensor_dict, image_tensor, images_dir, video_path, detect_results,
                              every_n_frames)

        with open(json_path, "w") as f:
            json.dump(detect_results, f, indent=2)

        make_xml(save_dir)

def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_dir", type=str, default='../video/REC-18',
        help="base directory path."
    )
    parser.add_argument(
        "save_dir", type=str, default='../v_out/REC-18',
        help="save directory path",
    )
    parser.add_argument(
        "--every_n_frames", "-n", type=int, default=5,
        help=""
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()
    detect_all(args.base_dir, args.save_dir, args.every_n_frames)
    # images = ['/media/aiteam/DATA/workspace/experiments/toppan-poc/v_out/REC-18/20180502131111/images/20180502132101_0105.jpg',
    # '/media/aiteam/DATA/workspace/experiments/toppan-poc/v_out/REC-18/20180502131111/images/20180502132101_0245.jpg']

    # img = []
    # for image_path in images:
    # image = cv2.imread('/DATA/workspace/thaopn/person_detection_frcnn/v_out/REC-18/20180503180000/images/20180503185500_0040.jpg')
    # image = Image.open('/DATA/workspace/thaopn/person_detection_frcnn/v_out/REC-18/20180503180000/images/20180503185500_0040.jpg')
    # im = image[:, :, ::-1].copy()
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # image_np = load_image_into_numpy_array(im)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # image_np_expanded = np.expand_dims(im, axis=0)
    # img.append(image)

    # run_inference_for_multi_image([im], detection_graph)

    # for i in images:
    #     # boxes = predic_image(i)
    #     im = cv2.imread(i)
    #     im = draw_boxes(im, boxes)
    #     while True:
    #         k = cv2.waitKey(30)
    #         if k == 27:
    #             break
    #         cv2.imshow('Image prediction', im)
    # cv2.destroyAllWindows()