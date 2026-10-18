"""ROS 2 node: subscriptions, EKF buffer and interpolation, params, TF and pose output (DOCK_FUSER_PIPELINE.md §3, §10)."""

import rclpy
from rclpy.node import Node


class DockFuserNode(Node):

    def __init__(self):
        super().__init__('dock_fuser')
        self.get_logger().info('dock_fuser scaffold: estimator not implemented yet')


def main(args=None):
    rclpy.init(args=args)
    node = DockFuserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
