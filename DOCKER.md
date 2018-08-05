docker build -t cumuluss/temporal-crop-classification:0.0.0 .

docker run -it -p 8888:8888 cumuluss/temporal-crop-classification:0.0.0 /bin/bash -c "jupyter notebook --ip='*' --port=8888 --no-browser --allow-root"

docker run -it cumuluss/temporal-crop-classification:0.0.0 /bin/bash 
