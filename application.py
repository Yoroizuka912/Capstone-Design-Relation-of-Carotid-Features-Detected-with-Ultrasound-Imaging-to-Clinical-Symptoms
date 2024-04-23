from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from utils import process_data, load_image, final_predict, generate_highlight, delete_files_with_uuid
from flask_cors import CORS
import json
import os
import uuid


UPLOAD_FOLDER = './static/temp'
STATIC_FOLDER = './static'
HOST = '127.0.0.1'
PORT = '5000'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'tif'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['HOST'] = HOST
app.config['PORT'] = PORT
CORS(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']
    mask = request.files['mask']
    file_uuid = str(uuid.uuid4())
    if image and allowed_file(image.filename) and mask and allowed_file(mask.filename):
        try:
            image_filename = secure_filename(image.filename)
            image_ext = os.path.splitext(image_filename)[-1]
            image_url = app.config['UPLOAD_FOLDER'] + "/" + file_uuid + '-image' + image_ext
            image.save(image_url)
            image_int = load_image(image_url)
            image_float = load_image(image_url, as_int=False)
            image_rgb = load_image(image_url, rgb=True)
        except:
            delete_files_with_uuid(app.config['UPLOAD_FOLDER'], file_uuid)
            return json.dumps({'code': 500, 'msg': 'Image read error!'})
        
        try:
            mask_filename = secure_filename(mask.filename)
            mask_ext = os.path.splitext(mask_filename)[-1]
            mask_url = app.config['UPLOAD_FOLDER'] + "/" + file_uuid + '-mask' + mask_ext
            mask.save(mask_url)
            mask = load_image(mask_url)
        except:
            delete_files_with_uuid(app.config['UPLOAD_FOLDER'], file_uuid)
            return json.dumps({'code': 500, 'msg': 'Mask read error!'})
        
        try:
            gender = request.form['gender']
        except:
            delete_files_with_uuid(app.config['UPLOAD_FOLDER'], file_uuid)
            return json.dumps({'code': 400, 'msg': 'Please specify gender!'})
        try:
            age = request.form['age']
        except:
            delete_files_with_uuid(app.config['UPLOAD_FOLDER'], file_uuid)
            return json.dumps({'code': 400, 'msg': 'Please specify age!'})
        history = request.form.getlist('history')
        
        data = process_data(history, gender)
        
        try:
            cnn_score, xgb_score, final_score = final_predict(image_int, image_float, file_uuid, mask, data)
        except:
            delete_files_with_uuid(app.config['UPLOAD_FOLDER'], file_uuid)
            return json.dumps({'code': 500, 'msg': 'Unknown error'})
        result = "Symptomatic" if final_score >= 0.5 else "Asymptomatic" 
        
        highlight_url = generate_highlight(image_rgb, mask, file_uuid)
        
        json_data = json.dumps({'code': 200, 'image_url': highlight_url, 'cnn_score': round(float(cnn_score), 3), 'xgb_score': round(float(xgb_score), 3), 'final_score': round(float(final_score), 3), 'result': result})
        
        delete_files_with_uuid(app.config['UPLOAD_FOLDER'], file_uuid)
        
        return json_data
    else:
        return json.dumps({'code': 400, 'msg': 'Please upload both images!'})
    
if __name__ == '__main__':
    app.run()
