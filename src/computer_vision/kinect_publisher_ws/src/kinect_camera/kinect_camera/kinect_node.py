import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import freenect
import cv2
import numpy as np

class KinectPublisher(Node):
        def __init__(self):
                super().__init__('kinect_publisher')
                self.bridge = CvBridge()
                self.rgb_pub = self.create_publisher(Image, 'kinect/rgb/image_raw', qos_profile=qos_profile_sensor_data)
                self.depth_pub = self.create_publisher(Image, 'kinect/depth/image_raw', qos_profile=qos_profile_sensor_data)
                self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        def timer_callback(self):

                rgb, _ = freenect.sync_get_video()
                depth, _ = freenect.sync_get_depth()


                if rgb is not None:
                        rgb_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), encoding='bgr8')
                        rgb_msg.header.stamp = self.get_clock().now().to_msg()
                        rgb_msg.header.frame_id = 'camera_link'
                        self.rgb_pub.publish(rgb_msg)
                        self.get_logger().info(f'Publishing frame RGB com timestamp: {rgb_msg.header.stamp}')

                if depth is not None:
                        depth_16 = depth.astype(np.uint16)
                        depth_msg = self.bridge.cv2_to_imgmsg(depth_16, encoding='mono16')
                        depth_msg.header.stamp = self.get_clock().now().to_msg()
                        depth_msg.header.frame_id = 'camera_link'
                        self.depth_pub.publish(depth_msg)
                        self.get_logger().info(f'Publishing frame Depth com timestamp: {depth_msg.header.stamp}')


def main(args=None):
        rclpy.init(args=args)
        logger = rclpy.logging.get_logger('main')

        node = KinectPublisher()

        logger.info("--- Main: Nó criado com sucesso. ---")

        logger.info("--- Main: Nó criado. Entrando no loop de spin do rclpy. ---")
        
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            logger.info("Nó interrompido pelo usuário (Ctrl+C).")
        finally:
            logger.info("Parando o loop síncrono do freenect...")
            freenect.sync_stop()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

if __name__ == '__main__':
        main()