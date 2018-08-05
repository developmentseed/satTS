# Docker Commands

docker build -t cumuluss/temporal-crop-classification:0.0.0 .

docker run -it cumuluss/temporal-crop-classification:0.0.0 /bin/bash 

docker run -it -p 8888:8888 cumuluss/temporal-crop-classification:0.0.0 /bin/bash -c "jupyter notebook --debug --ip=* --port=8888 --no-browser --allow-root"


# Docker Management

Running tsstack took about 50% memory limit of 5GB and 100% CPU
