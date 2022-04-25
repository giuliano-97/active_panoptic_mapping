from pathlib import Path

import numpy as np
import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from panoptic_mapping_msgs.msg import DetectronLabel, DetectronLabels
from pano_seg.predictor_factory import PredictorFactory
from pano_seg.constants import NYU40_IGNORE_LABEL, PANOPTIC_LABEL_DIVISOR, NYU40_THING_CLASSES
from pano_seg.visualization import colorize_panoptic_segmentation


def segments_info_to_labels_msg(segments_info) -> DetectronLabels:

    labels_msg = DetectronLabels()
    for sinfo in segments_info:
        if sinfo["category_id"] == 0:
            continue
        label = DetectronLabel()
        label.score = 0.5
        label.id = sinfo["id"]
        label.category_id = sinfo["category_id"]
        label.is_thing = sinfo["isthing"]
        if sinfo["isthing"]:
            label.instance_id = sinfo["id"]
        else:
            label.instance_id = sinfo["category_id"]
        labels_msg.labels.append(label)
    return labels_msg


class PanopticSegmentationNode:
    def __init__(self):
        # Init node
        rospy.init_node("pano_seg_node")

        # Load params
        self.visualize = rospy.get_param("~visualize", False)
        self.use_groundtruth = rospy.get_param("~use_groundtruth", False)
        if self.use_groundtruth:
            self.input_img_sub = message_filters.Subscriber("~input_image", Image)
            self.gt_instance_seg_sub = message_filters.Subscriber(
                "~gt_instance_seg", Image
            )
            self.gt_semantic_seg_sub = message_filters.Subscriber(
                "~gt_semantic_seg", Image
            )

            self.input_topics_ts = message_filters.TimeSynchronizer(
                [
                    self.input_img_sub,
                    self.gt_instance_seg_sub,
                    self.gt_semantic_seg_sub,
                ],
                queue_size=10,
            )
            self.input_topics_ts.registerCallback(
                self.input_image_and_gt_segmentation_cb
            )

        else:
            predictor_type = rospy.get_param("~predictor/type")
            model_dir_path = Path(rospy.get_param("~predictor/model_dir"))

            # Instantiate predictor
            self.predictor = PredictorFactory.get_predictor(
                predictor_type, model_dir_path, self.visualize
            )

            self.img_sub = rospy.Subscriber(
                "~input_image", Image, callback=self.input_image_cb
            )

        # Configure pano seg publisher
        self.cv_bridge = CvBridge()
        self.pano_seg_pub = rospy.Publisher("~pano_seg", Image, queue_size=100)
        self.labels_pub = rospy.Publisher("~labels", DetectronLabels, queue_size=100)

        if self.visualize:
            self.pano_seg_vis_pub = rospy.Publisher(
                "~pano_seg_vis", Image, queue_size=10
            )

    def input_image_and_gt_segmentation_cb(
        self,
        input_img_msg: Image,
        gt_instance_seg_msg: Image,
        gt_semantic_seg_msg: Image,
    ):
        gt_instance_seg = self.cv_bridge.imgmsg_to_cv2(gt_instance_seg_msg)
        gt_semantic_seg = self.cv_bridge.imgmsg_to_cv2(gt_semantic_seg_msg)

        gt_pano_seg = gt_semantic_seg * PANOPTIC_LABEL_DIVISOR + gt_instance_seg
        segments_info = []
        for id in np.unique(gt_pano_seg):
            category_id = id // PANOPTIC_LABEL_DIVISOR
            if category_id == NYU40_IGNORE_LABEL:
                gt_pano_seg[gt_pano_seg == id] = 0
                continue
            sinfo = {
                "id": id,
                "category_id": category_id,
                "isthing": category_id in NYU40_THING_CLASSES,
            }
            segments_info.append(sinfo)

        header = Header(stamp=input_img_msg.header.stamp, frame_id="depth_cam")
        pano_seg_msg = self.cv_bridge.cv2_to_imgmsg(gt_pano_seg.astype(np.uint16))
        pano_seg_msg.header = header
        labels_msg = segments_info_to_labels_msg(segments_info)
        labels_msg.header = header
        self.pano_seg_pub.publish(pano_seg_msg)
        self.labels_pub.publish(labels_msg)

        if self.visualize:
            pano_seg_vis, _ = colorize_panoptic_segmentation(gt_pano_seg)
            pano_seg_vis_msg = self.cv_bridge.cv2_to_imgmsg(pano_seg_vis)
            pano_seg_msg.header = header
            self.pano_seg_vis_pub.publish(pano_seg_vis_msg)

    def input_image_cb(self, img_msg: Image):
        image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        predictions = self.predictor(image)

        header = Header(stamp=img_msg.header.stamp, frame_id="depth_cam")

        pano_seg_msg = self.cv_bridge.cv2_to_imgmsg(predictions["panoptic_seg"])
        pano_seg_msg.header = header

        labels_msg = segments_info_to_labels_msg(predictions["segments_info"])
        labels_msg.header = header
        self.pano_seg_pub.publish(pano_seg_msg)
        self.labels_pub.publish(labels_msg)

        if self.visualize:
            pano_seg_vis_msg = self.cv_bridge.cv2_to_imgmsg(
                predictions["panoptic_seg_vis"]
            )
            pano_seg_vis_msg.header = header
            self.pano_seg_vis_pub.publish(pano_seg_vis_msg)


if __name__ == "__main__":
    pano_seg_node = PanopticSegmentationNode()
    rospy.loginfo("Predictor was successfully loaded.")
    rospy.spin()
