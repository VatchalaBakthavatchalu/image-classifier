# Image classifier Server (Python)

Server that runs a REST API which reads a publicly hosted image, and returns a classification. Accuracy is expected if category of image exists within imagenet, and if sufficient information clarity of category is present in the picture

## Installation

make sure you have the necessary dependencies, listed within ```setup.py```

## Usage

start server: ```python classifier-server.py```

curl the server, specifying the URL of the publicly hosted image within the data payload. 

Imagenet example curl:
```curl http://localhost:5000/classify/imagenet/img2 -d "data=https://3.bp.blogspot.com/-zBcdpq0NcLc/VrfAuIm_AzI/AAAAAAAAtCg/rymMidJo2-Y/s1600/individualImage.png" -X PUT```

MNIST example curl:
curl http://localhost:5000/classify/mnist/img2 -d "data=mnistimg.png" -X PUT
