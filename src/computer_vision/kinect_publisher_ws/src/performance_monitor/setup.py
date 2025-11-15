from setuptools import find_packages, setup

package_name = 'performance_monitor'

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
    description='Nó de monitoramento de performance para ROS 2',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'performance_monitor_node = performance_monitor.performance_monitor_node:main',
        ],
    },
)
