from flask import Flask, request, jsonify

# production
from app.torch_utils import transform_image, get_prediction
# dev
# from torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    # print('inside predict')
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            predicted_class_name = get_prediction(tensor)
            # print('here')
            # print(_)
            # print(prediction)
            # data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
            data = {'predicted class': predicted_class_name}
            return jsonify(data)
        except:
            # print(e)
            # print('here')
            return jsonify({'error': 'error during prediction'})
            # return jsonify({'error': f'{e} error during prediction'})