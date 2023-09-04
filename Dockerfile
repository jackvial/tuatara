FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
WORKDIR /workspaces/ocr-c++

# install generic tools
RUN apt update && apt -y dist-upgrade && \
apt install -y build-essential cmake gdb git git-lfs libssl-dev pkg-config unzip wget

# build and install opencv from source
# inspired from https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7
# image I/O libs
RUN apt install -y libjpeg-dev libpng-dev libtiff-dev
# video/audio Libs - FFMPEG, GSTREAMER, x264 and so on
RUN apt install -y libavcodec-dev libavformat-dev libswscale-dev
RUN apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt install -y libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev 
RUN apt install -y libfaac-dev libmp3lame-dev libvorbis-dev
# GTK lib for the graphical user functionalites coming from OpenCV highghui module
RUN apt install -y libgtk-3-dev
# Parallelism library C++ for CPU
RUN apt install -y libtbb-dev
# optimization libraries for OpenCV
RUN apt install -y libatlas-base-dev gfortran
# python libraries for python3:
RUN apt install -y python3.11-dev pybind11-dev
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py && rm get-pip.py
RUN pip3 install -U pip numpy Pillow

RUN apt clean

# clone opencv source 
RUN git clone https://github.com/opencv/opencv.git --depth 1 --branch 4.8.0 /workspaces/ocr-c++/opencv && \
    git clone https://github.com/opencv/opencv_contrib.git --depth 1 --branch 4.8.0 /workspaces/ocr-c++/opencv_contrib/

# build and install
RUN mkdir -p /workspaces/ocr-c++/opencv/build
WORKDIR /workspaces/ocr-c++/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_TBB=ON \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_CUBLAS=1 \
          -D WITH_CUDA=ON \
          -D BUILD_opencv_cudacodec=OFF \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D CUDA_ARCH_BIN="5.3 6.0 6.1 7.0 7.5 8.0 8.6" \
          -D CUDA_GENERATION=Auto \
          -D WITH_V4L=ON \
          -D WITH_QT=OFF \
          -D WITH_OPENGL=ON \
          -D WITH_GSTREAMER=ON \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D OPENCV_PC_FILE_NAME=opencv.pc \
          -D OPENCV_ENABLE_NONFREE=ON \
          -D OPENCV_EXTRA_MODULES_PATH=/workspaces/ocr-c++/opencv_contrib/modules \
          -D BUILD_opencv_python2=OFF \
          -D BUILD_opencv_python3=ON \
          -D PYTHON_VERSION=311 \
          -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3.11 \
          -D PYTHON3_EXECUTABLE=/usr/bin/python3.11 \
          -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
          -D PYTHON3_INCLUDE_DIR=/usr/include/python3.11 \
          -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.11/dist-packages \
          -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.11/dist-packages/numpy/core/include \
          -D BUILD_EXAMPLES=OFF \
          -D INSTALL_PYTHON_EXAMPLES=OFF \
          -D INSTALL_C_EXAMPLES=OFF \
          # -D BUILD_SHARED_LIBS=OFF \
          ..
RUN make -j6
RUN make install -j6

WORKDIR /workspaces/ocr-c++
RUN rm -rf /workspaces/ocr-c++/opencv && \
    rm -rf /workspaces/ocr-c++/opencv_contrib

# download libtorch
RUN mkdir -p /workspaces/ocr-c++/thirdparty
RUN wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip -O /workspaces/ocr-c++/thirdparty/libtorch.zip
RUN unzip /workspaces/ocr-c++/thirdparty/libtorch.zip -d /workspaces/ocr-c++/thirdparty/ && rm /workspaces/ocr-c++/thirdparty/libtorch.zip

# download models
RUN git clone https://huggingface.co/jackvial/tuatara-ocr-craft-and-parseq /workspaces/ocr-c++/models

# keep container running after start
ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]
