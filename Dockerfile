FROM developmentseed/geolambda:latest

RUN \
    yum makecache fast;
    
RUN pip3 install --upgrade pip
RUN pip3 install cython
RUN pip3 install pyyaml h5py 

ENV \
    PYCURL_SSL_LIBRARY=nss

# install requirements
WORKDIR /build
COPY requirements*txt /build/
RUN \
    pip3 install -r requirements.txt;
    #pip3 install -r requirements-dev.txt

# Jupyter and Tensorboard ports
EXPOSE 8888 6006

# Store notebooks in this mounted directory
VOLUME /notebooks

CMD ["/run_jupyter.sh"]

# install app
COPY . /build
RUN \
    pip3 install . -v; \
    rm -rf /build/*;

WORKDIR /home/geolambda
