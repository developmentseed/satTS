FROM developmentseed/geolambda:latest

ARG TENSORFLOW_VERSION=1.1.0 
ARG TENSORFLOW_ARCH=cpu
ARG KERAS_VERSION=2.2.0

RUN \
    yum makecache fast; \
    pip3 install cython; \
    pip3 install scipy; \
    pip3 install PyYAML; \
    pip3 install six; \
    pip3 install wheel; \

# Install TensorFlow
RUN pip --no-cache-dir install \
	https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl

# Install Keras
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}

ENV \
    PYCURL_SSL_LIBRARY=nss

# install requirements
WORKDIR /build
COPY requirements*txt /build/
RUN \
    pip3 install -r requirements.txt;
    #pip3 install -r requirements-dev.txt

# install app
COPY . /build
RUN \
    pip3 install . -v; \
    rm -rf /build/*;

WORKDIR /home/geolambda
