from flask import Flask, request, Blueprint, render_template
from flask_restful import Resource, Api
from mnistClassifier import MNIST_classifier
from imagenetClassifier import Imagenet_classifier

app = Flask(__name__)
api = Api(app)

simple_page = Blueprint('simple_page', __name__,
                        template_folder='templates')
@simple_page.errorhandler(404)
def page_not_found(e):
    return render_template('pages/404.html')

class ImageNetClassifier(Resource):    
    def put(self, img_path):
    	img_path = request.form['data'] 
    	imgnet = Imagenet_classifier()
    	classification = imgnet.predict_class(img_path)
    	image_net[img_path] = classification
        return {"image_classification": classification}

class MNISTClassifier(Resource):    
    def put(self, img_path):
    	img_path = request.form['data']
    	mnist = MNIST_classifier()
    	classification = mnist.predict_class(img_path)     
    	mnist_imgs[img_path] = classification
        return {"image_classification": classification}

api.add_resource(ImageNetClassifier, '/classify/imagenet/<string:img_path>')
api.add_resource(MNISTClassifier, '/classify/mnist/<string:img_path>')

if __name__ == '__main__':
    app.run(debug=True)