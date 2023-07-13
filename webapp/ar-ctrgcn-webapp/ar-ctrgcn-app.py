from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask import flash, request, redirect, url_for
import io
import os
from PIL import Image
import base64
import numpy as np
import pose_estimation_class as pm
from engineio.payload import Payload
import preprocessing.rt2_get_raw_skes_data
import preprocessing.rt2_get_raw_denoised_data
import preprocessing.rt2_seq_transformation
from werkzeug.utils import secure_filename
import cv2
import yaml

from recognition.classifier import Classifier

Payload.max_decode_packets = 2048

action_classes = ["drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw", "sitting down",
                    "standing up (from sitting position)", "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
                    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up",
                    "hand waving", "kicking something", "reach into pocket", "hopping (one foot jumping)", "jump up", "make a phone call/answer phone",
                    "playing with phone/tablet", "typing on a keyboard", "pointing to something with finger", "taking a selfie", "check time (from watch)",
                    "rub two hands together", "nod head/bow", "shake head", "wipe face", "salute", "put the palms together", "cross hands in front (say stop)",
                    "sneeze/cough", "staggering", "falling", "touch head (headache)", "touch chest (stomachache/heart pain)", "touch back (backache)",
                    "touch neck (neckache)", "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm", "punching/slapping other person",
                    "kicking other person", "pushing other person", "pat on back of other person", "point finger at the other person", "hugging other person",
                    "giving something to other person", "touch other person's pocket", "handshaking", "walking towards each other", "walking apart from each other",
                    "put on headphone", "take off headphone", "shoot at the basket", "bounce ball", "tennis bat swing", "juggling table tennis balls", "hush (quite)",
                    "flick hair", "thumb up", "thumb down", "make ok sign", "make victory sign", "staple book", "counting money", "cutting nails", "cutting paper (using scissors)",
                    "snapping fingers", "open bottle", "sniff (smell)", "squat down", "toss a coin", "fold paper", "ball up paper", "play magic cube",
                    "apply cream on face", "apply cream on hand back", "put on bag", "take off bag", "put something into a bag", "take something out of a bag",
                    "open a box", "move heavy objects", "shake fist", "throw up cap/hat", "hands up (both hands)", "cross arms", "arm circles", "arm swings",
                    "running on the spot", "butt kicks (kick backward)", "cross toe touch", "side kick", "yawn", "stretch oneself", "blow nose", "hit other person with something",
                    "wield knife towards other person", "knock over other person (hit with body)", "grab other person’s stuff", "shoot at other person with a gun",
                    "step on foot", "high-five", "cheers and drink", "carry something with other person", "take a photo of other person", "follow other person",
                    "whisper in other person’s ear", "exchange things with other person", "support somebody with hand", "finger-guessing game (playing rock-paper-scissors)"]

save_path = './'

UPLOAD_FOLDER = './static/uploads/'

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
        buffer.append(np.zeros((17, 4)))
        cnt += 1

    return buffer, cnt

def restore_buffer(buffer, cnt):
    """for i in range(cnt):
        buffer.pop()"""
    #clear buffer
    buffer = list()
    return buffer

def is_pred_time(frame_cnt, interval):
    result = False
    if((frame_cnt%interval) == 0):
        result = True
    return result

buffer, results, detector, frame_cnt, ef_inarow = None, None, None, None, None


@app.route('/', methods=['GET', 'POST'])
def index():
    global buffer, results, detector, frame_cnt, ef_inarow

    buffer = list()
    results = dict()
    frame_cnt = 0
    ef_inarow = 0

    return render_template('index.html')

   
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_video():
    if 'file' not in request.files:
        print("No file part")
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print("No image selected for uploading")
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Video successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)

@app.route('/display/<filename>')
def display_video(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)


@socketio.on('clear-buffer')
def clear_buffer(data):
    global buffer, frame_cnt, ef_inarow
    frame_cnt = 0
    ef_inarow = 0
    if buffer is not None:
        buffer.clear()
    else:
        buffer = list()
  
video_running = False

@socketio.on('start-video')
def start_video(video_name):
    global frame_cnt, buffer, detector, results, ef_inarow, video_running

    print("starting video")

    video_running = True
    video = cv2.VideoCapture(f'static/uploads/{video_name}')

    while video_running:
        success,frame = video.read()

        if success:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_img = np.array(Image.fromarray(img))
            
            relLmList = detector.getCoordinates(input_img)
            frame_cnt += 1

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

            #print('Keypoints detected: ', relLmList)

            if frame_cnt > 300:
                frame_cnt = 300

            # print('Frame count: ', frame_cnt)
                
            # Ogni 50 frame
            if is_pred_time(frame_cnt, 50):
                print('Prediction time!')
                #Riempio il buffer di liste vuote (300 frame richiesti per la classificazione)
                #buffer, n_empty_el = fill_buffer(buffer)

                #Faccio inferenza
                prediction, probabilities = predict(buffer)

                # Rimuovo i le liste vuote dal buffer o rimuovo i primi 100 se è pieno
                #buffer = restore_buffer(buffer, n_empty_el)
                if frame_cnt == 300:
                    frame_cnt = 200
                    buffer = buffer[100:]

                results['prediction'] = prediction
                results['probabilities'] = probabilities
                emit('results', results)
                #print('Results emitted', results)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

    video.release()

@socketio.on('stop-video')
def stop_video(data):
    global video_running
    video_running = False
    clear_buffer(None)


@socketio.on('image')
def image(data_image):

    global frame_cnt, buffer, detector, results, ef_inarow

    img = data_image.split(",")[1]
    imgdata = base64.b64decode(img)
    input_img = np.array(Image.open(io.BytesIO(imgdata)))
    
    #Estrazione keypoints
    frame_cnt += 1
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

    #print('Keypoints detected: ', relLmList)

    if frame_cnt > 300:
        frame_cnt = 300
        
    # Ogni 50 frame
    if is_pred_time(frame_cnt, 50):
        print('Prediction time!')
        #Riempio il buffer di liste vuote (300 frame richiesti per la classificazione)
        #buffer, n_empty_el = fill_buffer(buffer)

        #Faccio inferenza
        prediction, probabilities = predict(buffer)

        # Rimuovo i le liste vuote dal buffer o rimuovo i primi 100 se è pieno
        #buffer = restore_buffer(buffer, n_empty_el)

        if frame_cnt == 300:
            frame_cnt = 200
            buffer = buffer[100:]

        results['prediction'] = prediction
        results['probabilities'] = probabilities
        emit('results', results)



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
        skes_joints = preprocessing.rt2_seq_transformation.transform([denoised_skes_data])

    return skes_joints, success

  

if __name__ == "__main__":

    workdir = './models/ctrgcn'
    weights = './models/ctrgcn/runs-100-16300.pt'
    model_config = yaml.safe_load(open('./models/ctrgcn/config.yaml'))

    classifier1 = Classifier(workdir, weights, model_args=model_config['model_args'])

    classifier1.load_model()

    detector = pm.PoseDetector()

    socketio.run(app, host='0.0.0.0', debug=True)