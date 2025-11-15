import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from sensor_msgs.srv import SetCameraInfo
import yaml
import os

def save_camera_info(camera_info_msg, file_path):
    calib_data = {
        'image_width': camera_info_msg.width,
        'image_height': camera_info_msg.height,
        'camera_name': 'kinect_rgb',
        'camera_matrix': {'rows': 3, 'cols': 3, 'data': list(camera_info_msg.k)},
        'distortion_model': camera_info_msg.distortion_model,
        'distortion_coefficients': {'rows': 1, 'cols': 5, 'data': list(camera_info_msg.d)},
        'rectification_matrix': {'rows': 3, 'cols': 3, 'data': list(camera_info_msg.r)},
        'projection_matrix': {'rows': 3, 'cols': 4, 'data': list(camera_info_msg.p)}
    }
    with open(file_path, 'w') as file:
        yaml.dump(calib_data, file)

class CalibrationManager(Node):
    def __init__(self):
        super().__init__('calibration_manager')

        self.calibration_file = os.path.expanduser('~/.ros/camera_info/kinect_v1.yaml')
        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        
        self.camera_info_msg = self.load_camera_info()

        self.cam_info_pub = self.create_publisher(CameraInfo, 'kinect/rgb/camera_info', 10)
        self.srv = self.create_service(
            SetCameraInfo,
            '/kinect/rgb/set_camera_info',
            self.set_camera_info_callback)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info(f'Calibration Manager iniciado. Gerenciando: {self.calibration_file}')

    def load_camera_info(self):
        try:
            with open(self.calibration_file, "r") as file:
                # A ÚNICA MUDANÇA É AQUI: de safe_load para full_load
                calib_data = yaml.full_load(file)
            
            msg = CameraInfo()
            msg.width = calib_data["image_width"]
            msg.height = calib_data["image_height"]
            msg.k = [float(i) for i in calib_data["camera_matrix"]["data"]]
            msg.d = [float(i) for i in calib_data["distortion_coefficients"]["data"]]
            msg.r = [float(i) for i in calib_data["rectification_matrix"]["data"]]
            msg.p = [float(i) for i in calib_data["projection_matrix"]["data"]]
            msg.distortion_model = calib_data["distortion_model"]
            self.get_logger().info('Arquivo de calibração carregado com sucesso!')
            return msg
        except Exception as e:
            self.get_logger().warn(f'Arquivo de calibração não encontrado ou inválido ({type(e).__name__}). Aguardando calibração...')
            return None

    def set_camera_info_callback(self, request, response):
        self.get_logger().info('Nova calibração recebida pelo serviço.')
        self.camera_info_msg = request.camera_info
        
        try:
            save_camera_info(self.camera_info_msg, self.calibration_file)
            self.get_logger().info(f'Nova calibração salva em: {self.calibration_file}')
            response.success = True
            response.status_message = "Calibração salva com sucesso."
        except Exception as e:
            self.get_logger().error(f'Falha ao salvar a calibração: {e}')
            response.success = False
            response.status_message = f"Erro ao salvar: {e}"
        
        return response

    def timer_callback(self):
        # Se tivermos uma calibração carregada, publica-a
        if self.camera_info_msg:
            self.camera_info_msg.header.stamp = self.get_clock().now().to_msg()
            self.camera_info_msg.header.frame_id = 'camera_link'
            self.cam_info_pub.publish(self.camera_info_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()