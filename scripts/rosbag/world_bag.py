#!python
import csv
import rosbag
import rospy
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

with rosbag.Bag("coordinate_pf.bag", "w") as outbag:
    with rosbag.Bag(
        "/home/rd/code/mcmc-depth-images/data/p2_li/p2_li_25_50.bag", "r"
    ) as inbag:
        for topic, msg, t in inbag.read_messages(topics=["/tf", "/tf_static"]):
            outbag.write(topic, msg, t)
    with open("coordinate_pf.txt", "r") as csv_file:
        csv_reader = csv.DictReader(
            csv_file,
            fieldnames=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"],
            delimiter=" ",
        )
        for row in csv_reader:
            tf = TransformStamped()
            tf.header.stamp = rospy.Time.from_sec(float(row["timestamp"]))
            tf.header.frame_id = "camera_depth_optical_frame"
            tf.child_frame_id = "coordinate_pf"
            tf.transform.translation.x = float(row["tx"])
            tf.transform.translation.y = float(row["ty"])
            tf.transform.translation.z = float(row["tz"])
            tf.transform.rotation.x = float(row["qx"])
            tf.transform.rotation.y = float(row["qy"])
            tf.transform.rotation.z = float(row["qz"])
            tf.transform.rotation.w = float(row["qw"])
            tf_msg = TFMessage()
            tf_msg.transforms = [tf]
            outbag.write("/tf", tf_msg, tf.header.stamp)
