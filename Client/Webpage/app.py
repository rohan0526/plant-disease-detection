from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import os
import requests
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
DETECTRON_SERVER_VIS_URL = ''  # <ENTER YOUR Backend_LINK> Your Detectron server URL
DETECTRON_SERVER_URL1 = DETECTRON_SERVER_VIS_URL+'predictPotato'  # Your Detectron server prediction URL
DETECTRON_SERVER_URL2 = DETECTRON_SERVER_VIS_URL+'predictTomato' 

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the diagnosis page
@app.route('/disease_diagnosis')
def disease_diagnosis():
    return render_template('disease_diagnosis.html')

@app.route('/fertilizer_calculator')
def fertilizer_calculator():
    return render_template('fertilizer_calculator.html')

@app.route('/tomato_diagnosis', methods=['GET', 'POST'])
def tomato_diagnosis():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Change this to save the actual uploaded image
            file.save(filepath)

            # Delete previous visualization image if it exists
            vis_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'tomato_visualization.jpg')
            if os.path.exists(vis_filepath):
                os.remove(vis_filepath)

            # Send POST request to Detectron server
            with open(filepath, 'rb') as f:
                files = {'file': f}
                response = requests.post(DETECTRON_SERVER_URL2, files=files)

            if response.status_code == 200:
                result = response.json()

                disease_detected = result.get('disease_detected', [])

                # Handle visualization image
                visualization_url = result.get('visualization_url')
                if visualization_url:
                    vis_full_url = DETECTRON_SERVER_VIS_URL + visualization_url
                    vis_response = requests.get(vis_full_url, stream=True)
                    if vis_response.status_code == 200:
                        with open(vis_filepath, 'wb') as out_file:
                            shutil.copyfileobj(vis_response.raw, out_file)
                    del vis_response  # Cleanup response

                flash('Prediction successful!')
                return render_template('tomato_diagnosis.html', filename=filename,disease_detected=disease_detected, vis_exists=os.path.exists(vis_filepath))
            else:
                flash('Prediction failed. Please try again.')
                return redirect(request.url)
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect(request.url)
    else:
        return render_template('tomato_diagnosis.html')






# Route for the potato diagnosis page
@app.route('/potato_diagnosis', methods=['GET', 'POST'])
def potato_diagnosis():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Delete previous visualization image if it exists
            vis_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'visualization.jpg')
            if os.path.exists(vis_filepath):
                os.remove(vis_filepath)
            
            # Send POST request to Detectron server
            with open(filepath, 'rb') as f:
                files = {'file': f}
                response = requests.post(DETECTRON_SERVER_URL1, files=files)
            
            if response.status_code == 200:
                result = response.json()

                # Handle predicted classes
                predicted_classes = result.get('predicted_classes', [])

                # Handle visualization image
                visualization_url = result.get('visualization_url')
                if visualization_url:
                    vis_full_url = DETECTRON_SERVER_VIS_URL + visualization_url
                    vis_response = requests.get(vis_full_url, stream=True)
                    if vis_response.status_code == 200:
                        with open(vis_filepath, 'wb') as out_file:
                            shutil.copyfileobj(vis_response.raw, out_file)
                    del vis_response

                flash('Prediction successful!')
                return render_template('potato_diagnosis.html', filename=filename, result=predicted_classes)
            else:
                flash('Prediction failed. Please try again.')
                return redirect(request.url)
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
            return redirect(request.url)
    else:
        return render_template('potato_diagnosis.html')

# Route to display uploaded image
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True, port=8000)