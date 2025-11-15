import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/hallzero/Code/2025-1A-T01-G05-INTERNO/src/computer_vision/kinect_publisher_ws/install/kinect_camera'
