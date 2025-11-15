FROM ros:jazzy-ros-core

RUN apt update && apt upgrade -y && \
    apt install -y \ 
    build-essential \
    cmake \
    libusb-1.0-0-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-colcon-common-extensions \
    git \
    udev \
    ros-jazzy-cv-bridge \
    ros-jazzy-rtabmap-ros \
    nano \
    coreutils

RUN apt-get install -y freeglut3-dev libxmu-dev libxi-dev usbutils

RUN pip install setuptools --break-system-packages && \
    pip install --no-cache-dir cython numpy pynvml --break-system-packages

WORKDIR /libfreenect

RUN git clone https://github.com/OpenKinect/libfreenect && \
    cd libfreenect && \
    mkdir build && \
    cd build && \
    cmake .. \
          -DBUILD_EXAMPLES=ON \
          -DBUILD_PYTHON3=ON && \
    make && \
    make install && \
    ldconfig && \
    cp wrappers/python/python3/freenect*.so $(python3 -c "import site; print(site.getsitepackages()[0])") && \
    cd .. && \
    python3 ./src/fwfetcher.py && \
    mv ./audios.bin /usr/local/share/libfreenect/


ENV PYTHONPATH=/usr/local/lib/python3.12/site-packages:${PYTHONPATH:-}
ENV ROS_DOMAIN_ID=10
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

WORKDIR /

RUN rm -rf /libfreenect

SHELL ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && $0"]

WORKDIR /ros2_ws
RUN mkdir src

COPY kinect_publisher_ws/src /ros2_ws/src
RUN mkdir -p /root/.ros/camera_info/
COPY calibrationdata.tar.gz .
COPY kinect_initializer.py .
COPY kinect_v1.yaml /root/.ros/camera_info/kinect_v1.yaml

RUN colcon build --symlink-install

WORKDIR /ros2_ws