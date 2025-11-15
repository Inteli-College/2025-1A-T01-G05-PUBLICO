from setuptools import find_packages, setup

package_name = 'kinect_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hallzero',
    maintainer_email='filipikikuchi@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
		            'kinect_node = kinect_camera.kinect_node:main',
                    'calibration_manager = kinect_camera.calibration_manager:main'
        ],
    },
)
