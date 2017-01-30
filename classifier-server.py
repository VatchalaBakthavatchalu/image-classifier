from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from flask import Flask, request, Blueprint, render_template
from flask_restful import Resource, Api
import requests
import shutil


app = Flask(__name__)
api = Api(app)
imgs = {}

#example imagenet download: https://3.bp.blogspot.com/-zBcdpq0NcLc/VrfAuIm_AzI/AAAAAAAAtCg/rymMidJo2-Y/s1600/individualImage.png
def download_image(data):
	r = requests.get(data, stream=True)
	if r.status_code == 200:
	    with open("./current_img.jpg", 'wb') as f:
	        r.raw.decode_content = True
	        shutil.copyfileobj(r.raw, f)
	        img = image.load_img(f.name, target_size=(224,224))
	        img_arr = image.img_to_array(img)
	        return img_arr

def classify_image(img_path):
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

class ImageClassifier(Resource):    

    def put(self, img_path):
        imgs[img_path] = request.form['data']   
        classification = classify_image(imgs[img_path])     
        return {"image_classification": classification}

api.add_resource(ImageClassifier, '/<string:img_path>')

if __name__ == '__main__':
    app.run(debug=True)


