#!/usr/bin/env python3

import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

def publish_images_from_bag():
    # Initialize the ROS node
    rospy.init_node('camera', anonymous=True)

    # Define the publishers for the color and depth topics
    color_pub = rospy.Publisher('/device_0/sensor_1/Color_0/image/data', Image, queue_size=10)
    depth_pub = rospy.Publisher('/device_0/sensor_0/Depth_0/image/data', Image, queue_size=10)

    # Create a CvBridge object to convert OpenCV images to ROS Image messages
    bridge = CvBridge()

    # Open the rosbag
    bag = rosbag.Bag('/home/catkin_ws/src/detection/scripts/assets/outdoors.bag')

    # Create an iterator to loop through the bag messages continuously
    bag_messages = list(bag.read_messages())  # Read all messages in memory
    bag.close()  # Close the bag file, we'll re-open it in the loop if needed

    rate = rospy.Rate(30)  # Set the target frequency to 30 FPS

    while not rospy.is_shutdown():
        # Reopen the bag for each iteration to simulate continuous reading
        bag = rosbag.Bag('/home/catkin_ws/src/detection/scripts/assets/outdoors.bag')

        for topic, msg, t in bag_messages:
            rospy.loginfo("message sent")
            if topic == '/device_0/sensor_1/Color_0/image/data':
                color_pub.publish(msg)

            if topic == '/device_0/sensor_0/Depth_0/image/data':
                depth_pub.publish(msg)

            # Sleep to maintain a fixed 30 FPS (30 times per second)
            rate.sleep()

        bag.close()  # Close the bag file after replaying all messages

if __name__ == '__main__':
    try:
        publish_images_from_bag()
    except rospy.ROSInterruptException:
        pass
