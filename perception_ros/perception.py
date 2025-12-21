# cd ~/workspace/ros2_ws/ && colcon build && source install/setup.bash && ros2 launch gazebo_environment sonoma.launch.py
# colcon build && source install/local_setup.bash && ros2 run perception_ros perception
# colcon build && source install/local_setup.bash && ros2 run pure_pursuit visualizer

import rclpy # Import the ROS 2 client library for Python
from rclpy.node import Node # Import the Node class for creating ROS 2 nodes
from sensor_msgs.msg import Image,PointCloud2
from geometry_msgs.msg import PoseStamped,Pose


import cv2
from cv_bridge import CvBridge

import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import torch
from Perception.model.unet import UNet
from Perception.evaluate3 import UNetEvaluator

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from .rotation_utils import transform_pose

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

device = "cpu"
 
class Perception(Node):
    
 
    def __init__(self, trained_model="/home/ubuntu/workspace/ros2_ws/src/perception/perception-ros-homework-team-djs/Perception/model/epoch_39.pt"):
        super().__init__('Perception')
        
        # 1. Initialize the Evaluator ONCE. 
        # Pass your custom path to the weights here.
        self.unet = UNetEvaluator(weights_path=trained_model)

        self.image_subscription = self.create_subscription(
            Image,
            "/oakd/rgb/image_raw",
            self.image_callback,
            10)
            
        self.pose_publisher = self.create_publisher(PoseStamped, 'pose_msg', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.br = CvBridge()
        self.pose_msg = PoseStamped()
        
    def calculate_path(self, current_frame, pred):
        # 1. Get dimensions
        # pred is (180, 330)
        (h, w) = pred.shape
        
        # 2. Split mask into left and right halves
        left = pred[:, :w//2]
        right = pred[:, w//2:]

        # 3. Calculate Centers using NumPy (FAST)
        # Find coordinates where pixels are white (255)
        # We ignore the top 50 rows just like your original code
        l_coords = np.where(left[50:, :] > 0)
        r_coords = np.where(right[50:, :] > 0)

        # Calculate horizontal centers of mass
        if l_coords[1].size > 0:
            left_middle = np.mean(l_coords[1])
        else:
            left_middle = 0

        if r_coords[1].size > 0:
            # Remember to add the offset (w//2) for the right side
            right_middle = (w // 2) + np.mean(r_coords[1])
        else:
            right_middle = w - 1

        # Final target center
        middle = (right_middle + left_middle) / 2

        # 4. Visualization
        # Resize the background camera frame to match the prediction size (330x180)
        current_frame = cv2.resize(current_frame, (w, h), cv2.INTER_AREA)

        # Draw markers (Note: indices must be cast to int)
        lm, rm, mid = int(left_middle), int(right_middle), int(middle)
        
        current_frame[:, lm, 0] = 255   # Blue line for left lane
        current_frame[:, rm, 1] = 255   # Green line for right lane
        current_frame[:, mid, 2] = 255  # Red line for target center
        current_frame[:, w // 2, :] = 255 # White line for robot center

        cv2.imshow("UNet Perception Debug", current_frame)   
        cv2.waitKey(1)

        # 5. ROS2 Message Setup
        self.pose_msg.header.frame_id = "base_link"
        self.pose_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Physics logic: 
        # (Center of Image - Center of Lane) / Scale
        self.get_logger().info(f"w: {w}\nmiddle:{middle}\n")
        self.pose_msg.pose.position.x = 5.0
        self.pose_msg.pose.position.y = (float(w // 2) - middle) / 50.0 # Adjusted scale
        self.pose_msg.pose.position.z = 0.0
        self.pose_msg.pose.orientation.w = 1.0
        self.get_logger().info(f"error_y: {self.pose_msg.pose.position.y}")
        self.transform_and_publish_pose(self.pose_msg)


    def image_callback(self, msg: Image):
        self.get_logger().info("Received image")
        
        # Convert ROS msg to OpenCV (1080x720)
        current_frame = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # CORRECT USAGE:
        # The evaluator already has the model and transforms inside it.
        # Just pass the frame.
        pred = self.unet.evaluate(current_frame)
        # --- VISUALIZATION ---
        # Show the black and white mask
        cv2.imshow("UNet Prediction Mask", pred)
        
        # Important: waitKey(1) allows the window to refresh
        cv2.waitKey(1)
        # ---------------------
        # Run your math
        self.calculate_path(current_frame, pred)
    
    def transform_and_publish_pose(self,pose_msg : PoseStamped):
        '''
        try:
            t = self.tf_buffer.lookup_transform(
                "world",
                pose_msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            pose_msg.pose = transform_pose(pose_msg.pose, t)
            pose_msg.header.frame_id = "world"
            

        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {pose_msg.header.frame_id} to base_link: {ex}"
            )
            return
        '''
        
        self.get_logger().info("published_msg")

        self.pose_publisher.publish(self.pose_msg)

        
 
def main(args=None):
 
    rclpy.init(args=args)

    perception = Perception()
 

    rclpy.spin(perception)
 

    perception.destroy_node()
 

    rclpy.shutdown()
 
if __name__ == '__main__':

    main()
