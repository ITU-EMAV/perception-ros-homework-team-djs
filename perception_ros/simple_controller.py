# cd ~/workspace/ros2_ws/ && colcon build && source install/setup.bash && ros2 launch gazebo_environment sonoma.launch.py
# colcon build && source install/local_setup.bash && ros2 run perception_ros controller
# colcon build && source install/local_setup.bash && ros2 run pure_pursuit visualizer


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
import math

class Controller(Node):
    def __init__(self):
        super().__init__('controller')

    # 1. Subscribers and Publishers
        # Subscribes to the output of your Perception node
        self.subscription = self.create_subscription(
            PoseStamped,
            'pose_msg',
            self.listener_callback,
            10)
        
        # Publishes velocity commands to the vehicle
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

        # 2. Controller Parameters (Tuning)
        self.declare_parameter('linear_velocity', 6.25)  # Constant speed (m/s)
        self.declare_parameter('kp_steering', 0.7)      # Proportional gain for steering
        self.declare_parameter('kp_velocity', 5.0)      # Proportional gain for steering

        self.get_logger().info("Controller Node Started. Listening for pose_msg...")

    def listener_callback(self, msg: PoseStamped):
        """
        Processes the target pose and calculates steering commands.
        The perception node sends a 'world' frame pose, but for a simple
        proportional controller, we can look at the displacement.
        """
        
        # In your perception code, you set:
        # self.pose_msg.pose.position.y = (float(w // 2) - middle) / 50.0
        # This represents the lateral error.
        
        # Since the perception node already transformed this to the 'world' frame,
        # we ideally need the vehicle's current pose to find the relative error.
        # However, assuming 'pose_msg' effectively acts as a target offset:
        
        error_y = msg.pose.position.y
        
        self.get_logger().info(f"Heard error_y: {error_y}")

        # Initialize Twist message
        twist = Twist()

        # 1. Proportional Steering Logic
        # If error_y is positive, the target is to the left -> steer left (positive angular z)
        # If error_y is negative, the target is to the right -> steer right (negative angular z)
        kp_steering = self.get_parameter('kp_steering').get_parameter_value().double_value
        twist.angular.z = kp_steering * error_y

        # 2. Constant Linear Velocity
        base_vel = self.get_parameter('linear_velocity').get_parameter_value().double_value
        kp_velocity = self.get_parameter('kp_velocity').get_parameter_value().double_value
        twist.linear.x = base_vel / ((kp_velocity * error_y) + 1)


        # 3. Publish the command
        self.publisher_.publish(twist)

        self.get_logger().info(f"Published cmd_vel: Linear={twist.linear.x}, Angular={twist.angular.z:.2f}")

def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        # Stop the vehicle on shutdown
        stop_msg = Twist()
        controller.publisher_.publish(stop_msg)
        controller.get_logger().info("Shutting down... Vehicle stopped.")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()