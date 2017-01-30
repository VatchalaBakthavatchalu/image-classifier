from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from flask import Flask, request
from flask_restful import Resource, Api
import requests
import shutil


app = Flask(__name__)
api = Api(app)

imgs = {}
# data = "https://3.bp.blogspot.com/-zBcdpq0NcLc/VrfAuIm_AzI/AAAAAAAAtCg/rymMidJo2-Y/s1600/individualImage.png"

# https://3.bp.blogspot.com/-zBcdpq0NcLc/VrfAuIm_AzI/AAAAAAAAtCg/rymMidJo2-Y/s1600/individualImage.png
def download_image(data):
	r = requests.get(data, stream=True)
	if r.status_code == 200:
	    with open("./current_img.jpg", 'wb') as f:
	        r.raw.decode_content = True
	        shutil.copyfileobj(r.raw, f)
	        x = image.load_img(f.name, target_size=(224,224))
	        x = image.img_to_array(x)
	        return x
	        # return r.raw


def classify_image(img_path):
	# img_path = 'imagenet_example_beer.jpg'
	model = ResNet50(weights='imagenet')
	x = download_image(img_path)
	# x = image.load_img(img_path, target_size=(224, 224))
	# x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	preds = model.predict(x)
	# decode the results into a list of tuples (class, description, probability)
	# (one such list for each sample in the batch)
	# print('Predicted:', decode_predictions(preds, top=3)[0])
	# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)] 
	# print("prediction: " + str(decode_predictions(preds, top=1)))
	return decode_predictions(preds, top=1)[0][0][1]


# @simple_page.errorhandler(404)
# def page_not_found(e):
#     return render_template('pages/404.html')

class TodoSimple(Resource):    

    def put(self, img_path):
        imgs[img_path] = request.form['data']   
        prediction = classify_image(imgs[img_path])     
        # return {todo_id: todos[todo_id]}
        return {"image_classification": prediction}

api.add_resource(TodoSimple, '/<string:img_path>')

if __name__ == '__main__':
    app.run(debug=True)