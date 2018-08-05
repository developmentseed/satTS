FROM continuumio/anaconda3:latest

#### START -- From docker geolambda
ENV \
  PROJ_VERSION=5.1.0 \
  GEOS_VERSION=3.6.2 \
  HDF4_VERSION=4.2.12 \
  SZIP_VERSION=2.1.1 \
  HDF5_VERSION=1.10.1 \
    NETCDF_VERSION=4.6.1 \
  OPENJPEG_VERSION=2.3.0 \
    PKGCONFIG_VERSION=0.29.2 \
  GDAL_VERSION=2.3.1

# Paths to things
ENV \
  PREFIX=/usr/local \
  GDAL_CONFIG=/opt/conda/envs/tcc/bin/gdal-config

# install system libraries
RUN apt-get install -y wget tar gcc g++ zip rsync git ssh cmake bzip2 automake;
RUN apt-get clean;
#### END -- From docker geolambda

ENV CONDAENV_NAME=tcc

RUn pip install --upgrade pip

RUN conda update conda -y \
    && conda create -y -n $CONDAENV_NAME python=3.6 \
    gdal numpy matplotlib jupyter nb_conda

ENV PATH /opt/conda/envs/$CONDAENV_NAME/bin:$PATH

RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
  echo "conda activate" >> ~/.bashrc 

RUN [ "/bin/bash", "-c", "source activate $CONDAENV_NAME"]
RUN pip3 install gippy
