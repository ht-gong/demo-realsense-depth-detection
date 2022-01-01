#!/usr/bin/env python
# coding: utf-8
"""
Object Detection From TF2 Checkpoint
====================================
33

"""

# %%
# This demo will take you through the steps of running an "out-of-the-box" TensorFlow 2 compatible
# detection model on a collection of images. More specifically, in this example we will be using
# the `Checkpoint Format <https://www.tensorflow.org/guide/checkpoint>`__ to load the model.

# %%
# Download the test images
# ~~~~~~~~~~~~~~~~~~~~~~~~
# First we will download the images that we will use throughout this tutorial. The code snippet
# shown bellow will download the test images from the `TensorFlow Model Garden <https://github.com/tensorflow/models/tree/master/research/object_detection/test_images>`_
# and save them inside the ``data/images`` folder.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)



IMAGE_PATHS = ['image32', 'MicrosoftTeams-image1', 'MicrosoftTeams-image']

PATH_TO_MODEL_DIR = "models/my_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/"
PATH_TO_LABELS = "data/object_label_map.pbtxt"

# %%
# Load the model
# ~~~~~~~~~~~~~~
# Next we load the downloaded model
import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR

print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-8')).expect_partial()


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# %%
# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# %%
# Putting everything together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The code shown below loads an image, runs it through the detection model and visualizes the
# detection results, including the keypoints.
#
# Note that this will take a long time (several minutes) the first time you run this code due to
# tf.function's trace-compilation --- on subsequent runs (e.g. on new images), things will be
# faster.
#
# Here are some simple things to try out if you are curious:
#
# * Modify some of the input images and see if detection still works. Some simple things to try out here (just uncomment the relevant portions of code) include flipping the image horizontally, or converting to grayscale (note that we still expect the input image to have 3 channels).
# * Print out `detections['detection_boxes']` and try to match the box locations to the boxes in the image.  Notice that coordinates are given in normalized form (i.e., in the interval [0, 1]).
# * Set ``min_score_thresh`` to other values (between 0 and 1) to allow more detections in or to filter out more detections.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import pyrealsense2 as rs
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


"""
python3 -m object_detection.export_tflite_ssd_graph --pipeline_config_path models/my_training_model/pipeline.config --trained_checkpoint_prefix models/my_training_model/ckpt-8 --output_directory tflite_target/ --add_postprocessing_op=true

python3 -m object_detection.export_tflite_graph_tf2 --pipeline_config_path models/my_training_model/pipeline.config --trained_checkpoint_dir models/my_training_model/checkpoint --output_directory target_tflite/

python3 -m object_detection.export_tflite_graph_tf2 --pipeline_config_path training/pipeline.config --trained_checkpoint_dir training/checkpoint --output_directory target_tflite/

tflite_convert  --output_file=test.tflite --graph_def_file=saved_model.pb --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_dev_values=128 --input_shapes=1,300,300,3 --change_concat_input_ranges=false --allow_nudging_weights_to_use_fast_gemm_kernel=true --allow_custom_ops

"""
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

pipeline = rs.pipeline()

pipe_profile = pipeline.start(config)

# wait for camera to adjust exposure, etc.
for x in range(200):
    pipeline.wait_for_frames()

for it in range(10):
    frameset = pipeline.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()
    coloredpic = np.asanyarray(color_frame.get_data())

    # Align
    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    aligned_depth_frame = frameset.get_depth_frame()
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    depth = np.asanyarray(aligned_depth_frame.get_data())


    # Intrinsics & Extrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().get_intrinsics()
    color_intrin = color_frame.profile.as_video_stream_profile().get_intrinsics()
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
    color_to_depth_extrin = color_frame.profile.get_extrinsics_to(depth_frame.profile)

    print ("\n Depth intrinsics: " + str(depth_intrin))
    print ("\n Color intrinsics: " + str(color_intrin))
    print ("\n Depth to color extrinsics: " + str(depth_to_color_extrin))

    # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print ("\n\t depth_scale: " + str(depth_scale))
    print ("\n")

    start_time = time.time()
    image_np = coloredpic
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
    detections['num_detections'] = num_detections

        # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    boxes = detections['detection_boxes']
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = detections['detection_scores']
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh=.9
    # iterate over all objects found

    def get3dpoint(x_coord, y_coord):
        """
        Gets 3d-coordinate of pixel with (x_coord, y_coord) on colored frame

        Args:
          (x_coord, y_coord): coordinates of pixel

        Returns:
          [x, y, z]
        """
        pixel = [x_coord, y_coord]
        pixel_depth = depth[pixel[1], pixel[0]].astype(float)
        pixel_depth = pixel_depth * depth_scale
        point = rs.rs2_deproject_pixel_to_point(depth_intrin, pixel, pixel_depth)
        return point

    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        #
        if scores is None or scores[i] > min_score_thresh:
            # boxes[i] is the box which will be drawn
            # class_name = category_index[detections['detection_classes'][i]]['name']
            print("This box is gonna get used", boxes[i])
            (ymin, xmin, ymax, xmax) = boxes[i]
            xmin = round(xmin * color_frame.width)
            ymin = round(ymin * color_frame.height)
            xmax = round(xmax * color_frame.width)
            ymax = round(ymax * color_frame.height)
            print(xmin, ymin, xmax, ymax)

            print("Bounding box coords:")
            print(get3dpoint(xmin, ymin))
            print(get3dpoint(xmin, ymax))
            print(get3dpoint(xmax, ymin))
            print(get3dpoint(xmax, ymax))

            depthSum = 0
            validPixelCount = 0
            for x in range (xmin, xmax + 1):
                for y in range (ymin, ymax + 1):
                    pixelDepth = get3dpoint(x, y)[2]
                    if pixelDepth != 0:
                        depthSum += pixelDepth
                        validPixelCount += 1

            if validPixelCount < (ymax - ymin) * (xmax - xmin):
                print("Too much depth data missing for bounding box!")

            else:
                print("Avg depth over bounding box:")
                print(depthSum/validPixelCount)
            print("\n")




    viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes']+label_id_offset,
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.90,
    agnostic_mode=False)

    plt.figure()
    # plt.imshow(image_np_with_detections)
    plt.imsave('detected' + str(it) + '.png', image_np_with_detections)
    print('Done')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    plt.show()