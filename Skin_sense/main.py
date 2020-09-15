from flask import Flask, request, render_template#, Response
import numpy as np
import os
import string
import random

PATH = "C:/Users/Dell/Desktop/HackerEarth Healthcare/Project/others/"
OUTPUT_DIR = 'static'

app = Flask(__name__)

def generate_filename():
    return ''.join(random.choices(string.ascii_lowercase, k=20)) + '.jpg'

def get_prediction(image_path):
    import tensorflow as tf
    from model import SkinLesionTypeDetectionModel
    model = SkinLesionTypeDetectionModel(PATH+"model.json", PATH+"model.h5")
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    pred = model.predict_skin_lesion_type(image)
    
    print(pred)
    return pred

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            if uploaded_file.filename[-3:] in ['jpg', 'png']:
                image_path = os.path.join(OUTPUT_DIR, generate_filename())
                uploaded_file.save(image_path)
                class_name = "melanoma" #get_prediction(image_path)
                result = {
                    'class_name': class_name,
                    'path_to_image': image_path
                }
                return render_template('show.html', result=result)
    return render_template('index.html')

'''@app.route('/skin_lesion_image/<img>',methods=['POST'])
def skin_lesion_image(img):
    print("here")
    return Response(gen(img))'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,use_reloader=False)
