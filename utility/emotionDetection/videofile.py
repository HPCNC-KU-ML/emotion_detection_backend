import cv2
import sys
import numpy as np
from model import EMR
import json

# prevents opencl usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'scared']

data = {}
data['emotion'] = []


def format_image(image):
    """
    Function to format frame
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
            # determine whether the image is color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Image read from buffer
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    cascade = cv2.CascadeClassifier(
        "./utility/emotionDetection/haarcascade_frontalface_default.xml")
    # faces = cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    faces = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)

    if not len(faces) > 0:
        return [0]

    # initialize the first face as having maximum area, then find the one with max_area
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face

    # extract ROI of face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    try:
        # resize the image so that it can be passed to the neural network
        image = cv2.resize(
            image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("----->Problem during resize")
        return None

#     return image
    output = np.array(image)
    output = np.reshape(output, (48, 48, 1))

    return output


# Initialize object of EMR class
network = EMR()
network.build_network()

# name = input('Input videoname (abc.xyz) : ')
try:
    name = sys.argv[1]
    print(name)
except:
    print('please fill the videoname in input_video')
    exit()

cap = cv2.VideoCapture(name)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0
fps = cap.get(cv2.CAP_PROP_FPS)
duration = length/fps

sum = [0, 0, 0, 0, 0]

font = cv2.FONT_HERSHEY_SIMPLEX
feelings_faces = []

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


# Output file
out = cv2.VideoWriter('./result/video/output.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# append the list with the emoji images
for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))


while 1:
    ret, frame = cap.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    facecasc = cv2.CascadeClassifier(
        './utility/emotionDetection/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, 1.3, 5)
    new_frame = format_image(frame)

    # compute softmax probabilities
    if(len(new_frame) == 48):
        try:
            result = network.predict([new_frame])
        except:
            continue
    else:
        result = None

    # # compute softmax probabilities
    # if(len(new_frame) == 48):
    #     try:
    #         result = network.predict([format_image(frame)])
    #     except:
    #         continue
    # else:
    #     result = None

    if result is not None:
        # write the different emotions and have a bar to indicate probabilities for each class
        for index, emotion in enumerate(EMOTIONS):
            cv2.putText(frame, emotion, (10, index * 20 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (130, index * 20 + 10), (130 +
                                                          int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)

        # find the emotion with maximum probability and display it

        maxindex = np.argmax(result[0])
        if(result[0][maxindex] > float(sys.argv[2]) ):
            # print('hello')
            sum[maxindex] += 1
            # print(sum)

        if(frame_number % (int(fps/4)) == 0):
            maxindex2 = np.argmax(sum)
            sum = [0, 0, 0, 0, 0]
            print('time :{:.2f} emotion {}'.format(
                frame_number/fps, EMOTIONS[maxindex2]))

            # Add result to json file
            data['emotion'].append({
                'emotion': EMOTIONS[maxindex2],
                'time': str(frame_number/fps),
                'score': str(result[0][maxindex2])
            })

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, EMOTIONS[maxindex], (10, 360),
                    font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        face_image = feelings_faces[maxindex]

    if len(faces) > 0:
        # draw box around faces
        for face in faces:
            (x, y, w, h) = face
            frame = cv2.rectangle(
                frame, (x, y-30), (x+w, y+h+10), (255, 0, 0), 2)
            newimg = frame[y:y+h, x:x+w]
            newimg = cv2.resize(
                newimg, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
#             result = model.predict(newimg)
    out.write(frame)

    # cv2.imshow('Video', cv2.resize(frame, None, fx=2,
    #                                fy=2, interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

with open('./result/JSON/data.json', 'w') as outfile:
    json.dump(data, outfile)
