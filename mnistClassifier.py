import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


class MNIST_classifier():
	def __init__(self):
		self.model = Sequential()

		self.nb_classes = 10
		# input image dimensions
		self.img_rows, self.img_cols = 28, 28
		# number of convolutional filters to use
		self.nb_filters = 32
		# size of pooling area for max pooling
		self.pool_size = (2, 2)
		# convolution kernel size
		self.kernel_size = (3, 3)
		  
		self.input_shape = (self.img_rows, self.img_cols, 1)

		self.model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1],
		                        border_mode='valid',
		                        input_shape=self.input_shape))
		self.model.add(Activation('relu'))
		self.model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1]))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=self.pool_size))

		self.model.add(Flatten())
		self.model.add(Dense(128))
		self.model.add(Activation('relu'))
		self.model.add(Dense(self.nb_classes))
		self.model.add(Activation('softmax'))

		self.model.compile(loss='categorical_crossentropy',
		              optimizer='adadelta',
		              metrics=['accuracy'])

		self.model.load_weights("mnist-cnn-keras-model.hdf5", by_name=False)

	def predict_class(self, img_path):
		img = image.load_img(img_path, target_size=(self.img_rows,self.img_cols))
		img_arr = image.img_to_array(img)
		# print("array shape: " + str(img_arr.shape))
		# preprocess_img = img_arr.reshape(self.img_rows, self.img_cols, 1).astype('float32')
		img_arr /= 255
		processed_img = np.expand_dims(img_arr, axis=0)
		processed_img = np.swapaxes(processed_img, 0, 3)
		preds = self.model.predict_classes([processed_img])
		return str(preds[0])

