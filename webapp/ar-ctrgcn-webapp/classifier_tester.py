import cv2
import numpy as np
from recognition.classifier import Classifier
import pose_estimation_class as pm
import time
import preprocessing.rt2_get_raw_skes_data
import preprocessing.rt2_get_raw_denoised_data
import preprocessing.rt2_seq_transformation

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

def data_preprocessing(buffer):
    raw_data = dict()
    raw_data['num_frames'] = len(buffer)
    raw_data['num_actors'] = 2
    raw_data['actor1'] = dict()
    raw_data['actor1']['joints'] = buffer
    raw_data['actor1']['colors'] = []

    success = False

    # Preprocessing

    skes_data, success = preprocessing.rt2_get_raw_skes_data.get_raw_skes_data(raw_data)
    skes_joints = []

    if success:
        denoised_skes_data = preprocessing.rt2_get_raw_denoised_data.get_raw_denoised_data(skes_data)
        skes_joints = preprocessing.rt2_seq_transformation.transform(denoised_skes_data['joints'])

    #print('Data preprocessed, success:', success)

    return skes_joints, success

def predict(data):
    success = False
    top5_classes = ['Empty', 'Empty', 'Empty', 'Empty', 'Empty']
    output = [0, 0, 0, 0, 0]
    input_data, success = data_preprocessing(buffer)


    if success:
        top5_idx, output = classifier.eval(input_data)
        top5_classes = list()
        for id in top5_idx:
            top5_classes.append(action_classes[id])



    return top5_classes, output


if __name__ == '__main__':

    weights = 'recognition/csub/bone/runs-63-39438.pt'
    workdir = 'recognition/csub/bone/'
    classifier = Classifier(workdir, weights)
    classifier.load_model()

    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = pm.PoseDetector()
    frame_cnt = 0
    buffer = list()


    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue
        frame_cnt += 1
        #print(frame_cnt)
        lm = detector.getCoordinates(img)

        if len(lm) > 0:
            buffer.append(lm)
        else:
            buffer.append(list())

        if frame_cnt == 100 or frame_cnt == 200 or frame_cnt == 300:

            # Se il buffer non è pieno riempio di liste vuote (necessari 300 frame per la classificazione)
            if frame_cnt == 100 or frame_cnt == 200:
                r = 200 if frame_cnt == 100 else 100
                for i in range(r):
                    buffer.append(list())

            #print(len(buffer))
            prediction, probabilities = predict(buffer[:-300])

            #print('Top 5: ', prediction)
            #print('Prob: ', probabilities)

            # Rimuovo i le liste vuote dal buffer o rimuovo i primi 100 se è pieno
            if frame_cnt == 100:
                buffer = buffer[:-200]
            elif frame_cnt == 200:
                buffer = buffer[:-100]
            else:
                frame_cnt = frame_cnt - 100
                buffer = buffer[99:]

        if cv2.waitKey(1) == 27:
            break

        cv2.imshow('Test hand', img)

    cv2.destroyAllWindows()
    cap.release()