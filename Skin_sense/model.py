#import tensorflow as tf
from tensorflow.keras.models import model_from_json
#from tensorflow.python.keras.backend import set_session
import numpy as np
#import json
#import torch

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.15
#session = tf.compat.v1.Session(config=config)
#set_session(session)


class SkinLesionTypeDetectionModel(object):

    SKIN_LESION_TYPE_LIST = ['Actinic Keratoses','Basal Cell Carcinoma','Benign Keratosis',
  'Dermatofibroma','Melanoma','Melanocytic Nevi','Vascular skin lesion']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        #self.loaded_model.predict()
        
    def predict_skin_lesion_type(self, img):
        #global session
        #set_session(session)
        print(self.loaded_model)
        self.preds = self.loaded_model.predict(img)
        return SkinLesionTypeDetectionModel.SKIN_LESION_TYPE_LIST[np.argmax(self.preds)]

