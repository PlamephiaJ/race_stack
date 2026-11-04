ARG IMAGE=ros:humble
FROM ${IMAGE}
ARG USE_VIO=false

SHELL [ "/bin/bash", "-c" ]
ARG DEBIAN_FRONTEND=noninteractive

# Fix expired ROS 2 APT signing key
RUN set -eux; \
    rm -f \
        /etc/apt/sources.list.d/ros2.list \
        /etc/apt/sources.list.d/ros2-latest.list \
        /etc/apt/sources.list.d/ros2.sources; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl; \
    ROS_APT_SOURCE_VERSION="$( \
        curl -fsSL \
        https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -F '"tag_name"' \
        | cut -d '"' -f 4 \
    )"; \
    test -n "$ROS_APT_SOURCE_VERSION"; \
    . /etc/os-release; \
    ROS_UBUNTU_CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME}}"; \
    ROS_DEB_NAME="ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${ROS_UBUNTU_CODENAME}_all.deb"; \
    ROS_DEB_URL="https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/${ROS_DEB_NAME}"; \
    curl -fL --retry 3 \
        -o /tmp/ros2-apt-source.deb \
        "$ROS_DEB_URL"; \
    dpkg -i /tmp/ros2-apt-source.deb; \
    apt-get update; \
    rm -f /tmp/ros2-apt-source.deb

ADD ./.install_utils/basalt.repos /.install_utils/basalt.repos
ADD ./.install_utils/basalt_ros_setup.sh /.install_utils/basalt_ros_setup.sh
RUN if [[ "$USE_VIO" == "true" ]]; then chmod +x /.install_utils/basalt_ros_setup.sh && /.install_utils/basalt_ros_setup.sh; fi

ARG PLATFORM=arm
ARG USER
ARG UID
ARG GID

ADD ./.install_utils/realsense_setup.sh /.install_utils/realsense_setup.sh
RUN if [[ "$USE_VIO" == "true" ]]; then chmod +x /.install_utils/realsense_setup.sh && /.install_utils/realsense_setup.sh; fi


ADD ./.install_utils/linux_req/linux_req.txt /.install_utils/linux_req.txt
RUN if [[ "$PLATFORM" == "jet" ]]; then \
      mkdir -p /tmp/ocvdummy/DEBIAN && \
      printf '%s\n' \
        'Package: libopencv-dev' \
        'Version: 4.8.1' \
        'Architecture: all' \
        'Maintainer: none <none@localhost>' \
        'Provides: libopencv-core-dev, libopencv-imgproc-dev, libopencv-imgcodecs-dev, libopencv-videoio-dev, libopencv-highgui-dev, libopencv-calib3d-dev, libopencv-features2d-dev, libopencv-flann-dev, libopencv-ml-dev, libopencv-objdetect-dev, libopencv-photo-dev, libopencv-video-dev, libopencv-dnn-dev, libopencv-contrib-dev, libopencv-stitching-dev, libopencv-shape-dev, libopencv-superres-dev, libopencv-videostab-dev, libopencv-viz-dev, opencv-data' \
        'Description: Dummy, OpenCV 4.8.1 is provided by the base image' \
        > /tmp/ocvdummy/DEBIAN/control && \
      dpkg-deb --build /tmp/ocvdummy /tmp/libopencv-dev_4.8.1_all.deb && \
      dpkg -i /tmp/libopencv-dev_4.8.1_all.deb && \
      rm -rf /tmp/ocvdummy /tmp/libopencv-dev_4.8.1_all.deb; \
    fi

RUN apt-get update && xargs apt-get install -y < /.install_utils/linux_req.txt
COPY ./.install_utils/python_req_portable.txt /tmp/python_req_portable.txt

# Ubuntu 22.04 provides the compatible SciPy and Matplotlib versions
# on ARM64, avoiding unsupported source builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev python3-scipy python3-matplotlib

# Never suppress dependency installation failures.
RUN python3 -m pip install \
    --index-url https://pypi.org/simple \
    -r /tmp/python_req_portable.txt

# Preserve NVIDIA's OpenCV on Jetson; install OpenCV on non-Jetson hosts.
RUN if [[ "${PLATFORM}" != "jet" ]]; then \
      python3 -m pip install \
        --index-url https://pypi.org/simple \
        "opencv-python==4.8.1.78"; \
    fi

# Install quadprog explicitly because trajectory-planning-helpers has
# an older, incompatible quadprog dependency declaration.
RUN python3 -m pip install \
      --index-url https://pypi.org/simple \
      "setuptools==58.2.0" wheel "cython<3" && \
    SETUPTOOLS_USE_DISTUTILS=stdlib \
    python3 -m pip install \
      --index-url https://pypi.org/simple \
      --no-build-isolation \
      "quadprog==0.1.13" && \
    python3 -m pip install \
      --index-url https://pypi.org/simple \
      --no-deps \
      "trajectory-planning-helpers==0.79"

# Install the exact optimizer source tracked as a Git submodule.
# Missing submodules must fail the Docker build.
COPY planner/global_planner/global_planner/global_racetrajectory_optimization /tmp/race_stack_optimizer

RUN test -f /tmp/race_stack_optimizer/setup.py && \
    python3 -m pip install \
      --no-deps --no-build-isolation \
      /tmp/race_stack_optimizer && \
    python3 -c "import global_racetrajectory_optimization"

ADD ./.install_utils /.install_utils
RUN useradd -ms /bin/bash -u ${UID} -G sudo ${USER} && groupadd -fg ${GID} ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USER

ADD .install_utils/bashrc/ /tmp/.install_utils/
RUN echo "export USERNAME=$USER" >> /home/$USER/.bashrc &&\
    cat /tmp/.install_utils/.bashrc_git >> /home/$USER/.bashrc &&\
    cat /tmp/.install_utils/.bashrc_forzaeth >> /home/$USER/.bashrc

USER root
# Clean up
RUN rm -rf /tmp/.install_utils

USER $USER
WORKDIR /home/$USER/ws
CMD ["/bin/bash"]
