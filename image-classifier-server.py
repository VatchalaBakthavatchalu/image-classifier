from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from flask import Flask, request, Blueprint, render_template
from flask_restful import Resource, Api
import requests
import shutil
from mnistClassifier import MNIST_classifier


app = Flask(__name__)
api = Api(app)
imgs = {}
mnist_imgs = {}

#example imagenet download: https://3.bp.blogspot.com/-zBcdpq0NcLc/VrfAuIm_AzI/AAAAAAAAtCg/rymMidJo2-Y/s1600/individualImage.png
#example mnist download:  
def download_image(data):
	r = requests.get(data, stream=True)
	if r.status_code == 200:
	    with open("./current_img.jpg", 'wb') as f:
	        r.raw.decode_content = True
	        shutil.copyfileobj(r.raw, f)
	        img = image.load_img(f.name, target_size=(224,224))
	        img_arr = image.img_to_array(img)
	        return img_arr

def classify_imagenet(img_path):
	model = ResNet50(weights='imagenet')
	img_arr = download_image(img_path)
	img_arr = np.expand_dims(img_arr, axis=0)
	processed_img = preprocess_input(img_arr)
	preds = model.predict(processed_img)
	#decoded prediction output format example: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), ...
	return decode_predictions(preds, top=1)[0][0][1]

simple_page = Blueprint('simple_page', __name__,
                        template_folder='templates')
@simple_page.errorhandler(404)
def page_not_found(e):
    return render_template('pages/404.html')

class ImageNetClassifier(Resource):    

    def put(self, img_path):
        if img_path in imgs.keys():
        	classification = imgs[img_path]
        else:
        	imgs[img_path] = request.form['data']   
        	classification = classify_image(imgs[img_path])     
        	imgs[img_path] = classification
        return {"image_classification": classification}

class MNISTClassifier(Resource):    

    def put(self, img_path):
        if img_path in mnist_imgs.keys():
        	classification = mnist_imgs[img_path]
        else:
        	img_path = request.form['data'] 
        	print("img path: " + img_path)
        	mnist = MNIST_classifier()
        	classification = mnist.predict_class(img_path)     
        	mnist_imgs[img_path] = classification
        return {"image_classification": classification}

api.add_resource(ImageNetClassifier, '/classify/imagenet/<string:img_path>')
api.add_resource(MNISTClassifier, '/classify/mnist/<string:img_path>')
#TODO: add batch endpoints

if __name__ == '__main__':
    app.run(debug=True)

99