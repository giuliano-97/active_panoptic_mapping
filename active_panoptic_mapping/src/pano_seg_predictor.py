from pathlib import Path
import queue

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from pano_seg.predictor_factory import PredictorFactory
from panoptic_mapping_msgs.msg import DetectronLabel, DetectronLabels


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
        predictor_type = rospy.get_param("~predictor/type")
        model_dir_path = Path(rospy.get_param("~predictor/model_dir"))

        # Instantiate predictor
        self.predictor = PredictorFactory.get_predictor(predictor_type, model_dir_path, self.visualize)

        # Configure pano seg publisher
        self.cv_bridge = CvBridge()
        self.pano_seg_pub = rospy.Publisher("~pano_seg", Image, queue_size=100)
        self.labels_pub = rospy.Publisher("~labels", DetectronLabels, queue_size=100)

        if self.visualize:
            self.pano_seg_vis_pub = rospy.Publisher("~pano_seg_vis", Image, queue_size=100)

        # Configure image subscriber
        self.img_sub = rospy.Subscriber("~input_image", Image, callback=self.predict_cb)

    def predict_cb(self, img_msg: Image):
        image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        predictions = self.predictor(image)

        header = Header(stamp=rospy.Time.now(), frame_id="depth_cam")

        pano_seg_msg = self.cv_bridge.cv2_to_imgmsg(predictions["panoptic_seg"])
        pano_seg_msg.header = header

        labels_msg = segments_info_to_labels_msg(predictions["segments_info"])
        labels_msg.header = header
        self.pano_seg_pub.publish(pano_seg_msg)
        self.labels_pub.publish(labels_msg)

        if self.visualize:
            pano_seg_vis_msg = self.cv_bridge.cv2_to_imgmsg(predictions["panoptic_seg_vis"])
            pano_seg_vis_msg.header = header
            self.pano_seg_vis_pub.publish(pano_seg_vis_msg)


if __name__ == "__main__":
    pano_seg_node = PanopticSegmentationNode()
    rospy.spin()
