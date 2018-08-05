docker build -t cumuluss/temporal-crop-classification:0.0.0 . -f Dockerfile.tcc

docker run -it -p 8888:8888 cumuluss/temporal-crop-classification:0.0.0 /bin/bash -c "mkdir /opt/notebooks && jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root"
