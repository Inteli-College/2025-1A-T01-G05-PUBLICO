import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class KinectSubscriber(Node):
    def __init__(self):
        super().__init__('kinect_subscriber')
        self.bridge = CvBridge()
        self.rgb_sub = self.create_subscription(Image, "image_raw", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, "depth/image_raw", self.depth_callback, 10)
    
    def rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("RGB Image", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Erro na conversão RGB: {e}")

    def depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            depth_display = cv2.convertScaleAbs(cv_image, alpha=0.03)  # para visualização
            cv2.imshow("Depth Image", depth_display)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Erro na conversão de profundidade: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = KinectSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
    