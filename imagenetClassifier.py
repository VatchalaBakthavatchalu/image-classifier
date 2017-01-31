from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import requests
import shutil


class Imagenet_classifier():
	#example imagenet download: https://3.bp.blogspot.com/-zBcdpq0NcLc/VrfAuIm_AzI/AAAAAAAAtCg/rymMidJo2-Y/s1600/individualImage.png
	#example mnist download:  
	def download_image(self, data):
		r = requests.get(data, stream=True)
		if r.status_code == 200:
		    with open("./current_img.jpg", 'wb') as f:
		        r.raw.decode_content = True
		        shutil.copyfileobj(r.raw, f)
		        img = image.load_img(f.name, target_size=(224,224))
		        img_arr = image.img_to_array(img)
		        return img_arr

	def predict_class(self, img_path):
		model = ResNet50(weights='imagenet')
		img_arr = self.download_image(img_path)
		img_arr = np.expand_dims(img_arr, axis=0)
		processed_img = preprocess_input(img_arr)
		preds = model.predict(processed_img)
		#decoded prediction output format example: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), ...
		return decode_predictions(preds, top=1)[0][0][1]