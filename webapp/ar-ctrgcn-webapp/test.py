from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import io
from PIL import Image
import base64
import numpy as np
import pose_estimation_class as pm
from engineio.payload import Payload
import preprocessing.rt2_get_raw_skes_data
import preprocessing.rt2_get_raw_denoised_data
import preprocessing.rt2_seq_transformation
import os
from flask import flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from recognition.classifier import Classifier

Payload.max_decode_packets = 2048

#device = torch.device("cpu")
#n_gpu = torch.cuda.device_count()
#torch.cuda.get_device_name(0)
    
action_classes = ['Drink water', 'Eating', 'Brushing teeth', 'Brushing hair', 'Drop', 'Pickup', 'Throw', 'Sitting down', 'Standing up', 'Clapping',
                  'Reading', 'Writing', 'Tear up paper', 'Wear jacket', 'Take off jacket', 'Wear a shoe', 'Take off a shoe', 'Wear on glasses', 'Take off glasses', 'Put on a hat/cap',
                  'Take off a hat/cap', 'Cheer up', 'Hand waving', 'Kicking something', 'Reach into pocket', 'Hopping', 'Jump up', 'Make a phone call', 'Playing with phone', 'Typing on a keyboard',
                  'Pointing to something with finger', 'Taking a selfie', 'Check time (from watch)', 'Rub two hands together', 'Nod head/bow', 'Shake head', 'Wipe face', 'Salute', 'Put the palms together', 'Cross hands in front (say stop)',
                  'Sneeze/cough', 'Staggering', 'Falling', 'Touch head (headache)', 'Touch chest (stomachache/heart pain)', 'Touch back (backache)', 'Touch neck (neckache)', 'Nausea or vomiting condition', 'Use a fan (with hand or paper)/feeling warm', 'Punching/slapping other person',
                  'Kicking other person', 'Pushing other person', 'Pat on back of other person', 'Point finger at the other person', 'Hugging other person', 'Giving something to other person', "Touch other person's pocket", 'Handshaking', 'Walking towards each other', 'Walking apart from each other'
                  ]



save_path = './'

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)   
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
socketio = SocketIO(app, cors_allowed_origins='*', ping_timeout=500000, max_http_buffer_size=50000000)

BUFFER_SIZE = 300
empty_frame_treshold = 50



def fill_buffer(buffer):
    cnt = 0
    if len(buffer) > BUFFER_SIZE:
        buffer = buffer[:BUFFER_SIZE]

    while(len(buffer) < BUFFER_SIZE):
        buffer.append(list())
        cnt += 1

    return buffer, cnt

def restore_buffer(buffer, cnt):
    for i in range(cnt):
        buffer.pop()
    return buffer

def is_pred_time(frame_cnt, interval):
    result = False
    if((frame_cnt%interval) == 0):
        result = True
    return result


global buffer, results, detector, frame_cnt, ef_inarow



@app.route('/', methods=['GET', 'POST'])
def index():
    global buffer, results, detector, frame_cnt, ef_inarow

    buffer = list()
    results = dict()
    detector = pm.PoseDetector()
    frame_cnt = 0
    ef_inarow = 0

    return render_template('index.html')



@socketio.on('clear-buffer')
def clear_buffer(data):
    global buffer, frame_cnt, ef_inarow
    frame_cnt = 0
    ef_inarow = 0
    buffer.clear()
  


@socketio.on('image')
def image(data_image):

    global frame_cnt, buffer, detector, results, ef_inarow

    img = data_image.split(",")[1]
    imgdata = base64.b64decode(img)
    input_img = np.array(Image.open(io.BytesIO(imgdata)))

    #Estrazione keypoints
    frame_cnt += 1
    print("F:", frame_cnt, "buffer len:", len(buffer))
    relLmList = detector.getCoordinates(input_img)

    if len(relLmList) < 1:
        ef_inarow += 1
        buffer.append(relLmList)
        if ef_inarow > 40:
            buffer.clear()
            frame_cnt = 0
            ef_inarow = 0
    else:
        ef_inarow = 0
        buffer.append(relLmList)

    print('Keypoints detected: ', relLmList)

    if frame_cnt > 1:
        frame_cnt = 300

    # Ogni 50 frame
    if is_pred_time(frame_cnt, 50):
        #Riempio il buffer di liste vuote (300 frame richiesti per la classificazione)
        buffer, n_empty_el = fill_buffer(buffer)
    
        #Faccio inferenza
        prediction, probabilities = predict(buffer)
        
        print(prediction)
        print(probabilities)

        # Rimuovo i le liste vuote dal buffer o rimuovo i primi 100 se è pieno
        buffer = restore_buffer(buffer, n_empty_el)
    
        if frame_cnt == 300:
            frame_cnt = 200
            buffer = buffer[100:]
    
        results['prediction'] = prediction
        results['probabilities'] = probabilities
        emit('results', results)
        print('Results emitted')



@socketio.on('imageobj')
def imageobj(data_image):
    pass



@app.route('/upload')
def upload_form():
	return render_template('index.html')
    
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_video():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	else:
		filename = secure_filename(file.filename)
		#file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_video filename: ' + filename)
		flash('Video successfully uploaded and displayed below')
		return render_template('index.html', filename=filename)

@app.route('/display/<filename>')
def display_video(filename):
	#print('display_video filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

def predict(buffer):
    success = False
    top5_classes = ['Empty', 'Empty', 'Empty', 'Empty', 'Empty']
    accuracies = [0, 0, 0, 0, 0]

    input_data, success = data_preprocessing(buffer)

    if success:
        top5_idx, output = classifier1.eval(input_data)
        top5_classes = list()
        accuracies = list()


        for id in top5_idx:
            top5_classes.append(action_classes[id])
        for o in output:
            accuracies.append(round(o*100, 2))
            
    return top5_classes, accuracies


def data_preprocessing(buffer):

    raw_data = dict()
    raw_data['num_frames'] = len(buffer)
    raw_data['num_actors'] = 1
    raw_data['actor1'] = dict()
    raw_data['actor1']['joints'] = buffer


    success = False

    # Preprocessing

    skes_data, success = preprocessing.rt2_get_raw_skes_data.get_raw_skes_data(raw_data)
    skes_joints = []

    if success:
        denoised_skes_data = preprocessing.rt2_get_raw_denoised_data.get_raw_denoised_data(skes_data)
        skes_joints = preprocessing.rt2_seq_transformation.transform(denoised_skes_data['joints'])

    return skes_joints, success

def data_preprocessing_obj(buffer):

    raw_data = dict()
    raw_data['num_frames'] = len(buffer)
    raw_data['num_actors'] = 1
    raw_data['actor1'] = dict()
    raw_data['actor1']['joints'] = buffer

    success = False

    # Preprocessing

    skes_data, success = preprocessing.rt3_get_raw_skes_data.get_raw_skes_data(raw_data)
    skes_joints = []

    if success:
        denoised_skes_data = preprocessing.rt3_get_raw_denoised_data.get_raw_denoised_data(skes_data)
        skes_joints = preprocessing.rt3_seq_transformation.transform(denoised_skes_data['joints'])

    return skes_joints, success
  

if __name__ == "__main__":

    weights1 = './ctrgcn/model.pth'
    workdir1 = './ctrgcn'
      
    classifier1 = Classifier(workdir1, weights1)

    classifier1.load_model()

    socketio.run(app, host='0.0.0.0', debug=True)
