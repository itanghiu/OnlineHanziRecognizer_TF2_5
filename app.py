from flask import Flask, render_template, send_from_directory
from flask import request
import base64
from PIL import Image
from io import BytesIO
import Cnn
import numpy
import Utils
import logging
import json
import tensorflow as tf
from Cnn import Cnn
from Utils import preprocess_image

logger = logging.getLogger('app.py')
logging.basicConfig(filename='webApp.log', level=logging.DEBUG)
app = Flask(__name__)
cnn =Cnn()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)

@app.route('/addCharImage/', methods=['POST'])
def add_char_image():
        charBase64 = request.get_json().get("value")
        image = Image.open(BytesIO(base64.b64decode(charBase64))) #converts from base64 to png
        hand_written_char_file_name = Utils.HAND_WRITTEN_CHAR_FILE_NAME
        image.save(hand_written_char_file_name, 'PNG')
        # image is RGB
        image = Utils.set_image_background_to_white(image)
        image = Utils.crop_image(image)
        image.save(hand_written_char_file_name, 'PNG')
        predicted_chars, predicted_probabilities, indices = cnn.recognize_image(hand_written_char_file_name)
        #predicted_probabilities = list(map(str, predicted_probabilities)) # converts float list to string list
        predicted_probabilities = [round(proba, 2) for proba in predicted_probabilities]
        predicted_probabilities = list(map(str, predicted_probabilities))  # converts float list to string list
        logger.info('Predicted chars: ' + ":".join(predicted_chars))
        logger.info('Predicted probabilities: ' + ", ".join(predicted_probabilities))
        result = dict(chars=predicted_chars, probabilities=predicted_probabilities)
        jsonResult = json.dumps(result)
        return jsonResult


@app.route('/getCharImage/', methods=['GET'])
def get_char_image():
        return "Hi everyone !"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False)
    cnn.tf.app.run()