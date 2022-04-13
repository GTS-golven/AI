# Import the necessary packages
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder
import cv2


def object_detection(model_dir, videocapture):
    PATH_TO_CFG = os.path.join(model_dir, 'pipeline.config')
    PATH_TO_LABELS = os.path.join(model_dir, 'label_map.pbtxt')

    # Load the pretrained model
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    tf.get_logger().setLevel('ERROR')

    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'checkpoint', 'ckpt-0')).expect_partial()

    def get_model_detection_function(model):
        """Get a tf.function for detection."""

        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            detections = model.postprocess(prediction_dict, shapes)
            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn

    detect_fn = get_model_detection_function(detection_model)

    # Load label map data (for plotting)
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

    # Define the video stream
    if videocapture == '0':
        cap = cv2.VideoCapture(0)

    else:
        cap = cv2.VideoCapture(videocapture)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('j', 'p', 'e', 'g'), 10, (800, 600))

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

    while True:

        # Read frame from camera or video file
        ret, image_np = cap.read()

        # If image you need to load into numPy array manually
        if videocapture != '0':
            image_np = load_image_into_numpy_array(videocapture)

        input_tensor = tf.convert_to_tensor(np.expand_dims(cv2.resize(image_np, (400, 300)), 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        # Display output
        # cv2.imshow('object detection', cv2.resize(image_np_with_detections, (400, 300)))
        out.write(image_np_with_detections)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
