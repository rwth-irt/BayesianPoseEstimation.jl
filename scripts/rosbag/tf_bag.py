#!python
import csv
import os
import rosbag
import rospy
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

experiment = "p2_li_25_50"
configuration = "coordinate_pf"
data_dir = "/home/rd/code/mcmc-depth-images/data"
exp_pro = os.path.join(data_dir, "exp_pro", "pf", experiment)
exp_raw = os.path.join(data_dir, "exp_raw", "pf", experiment)

original_bag = os.path.join(data_dir, "rosbags", experiment, "original.bag")
pf_tum = os.path.join(exp_raw, configuration + ".tum")
tf_bag = os.path.join(exp_pro, configuration + ".bag")

if not os.path.exists(exp_pro):
    # Create a new directory because it does not exist
    os.makedirs(exp_pro)

with rosbag.Bag(tf_bag, "w") as outbag:
    with rosbag.Bag(original_bag, "r") as inbag:
        for topic, msg, t in inbag.read_messages(topics=["/tf", "/tf_static"]):
            outbag.write(topic, msg, t)
    with open(pf_tum, "r") as csv_file:
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
