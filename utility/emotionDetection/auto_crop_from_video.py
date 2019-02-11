
import sys
import face_recognition
import cv2
import os


current_path = os.getcwd()

# Open the input movie file
video_name = input('input video name in video dir (sam1.mp4) : ')
video_name
video_des = 'input_video/'+video_name

input_movie = cv2.VideoCapture(video_des)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

char1 = input('1st charactor picture file(ester.jpg) : ')
char2 = input('2nd charactor picture file(bie.jpg) : ')

char_des1 = './charactor/' + char1
char_des2 = './charactor/' + char2

name1 = char1.split('.')[0]
name2 = char2.split('.')[0]
if not os.path.exists("./face_database/" + name2):
    os.mkdir("./face_database/" + name2)
if not os.path.exists("./face_database/" + name1):
    os.mkdir("./face_database/" + name1)


# Load some sample pictures and learn how to recognize them.
lmm_image = face_recognition.load_image_file(char_des1)
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

al_image = face_recognition.load_image_file(char_des2)
al_face_encoding = face_recognition.face_encodings(al_image)[0]


known_faces = [
    lmm_face_encoding,
    al_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0


count = 0
count1 = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break
    if(frame_number % 1 == 0):
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(
                known_faces, face_encoding, tolerance=0.50)

            # If you had more than 2 faces, you could make this logic a lot prettier
            # but I kept it simple for the demo
            name = None
            if match[0]:
                name = name1
            elif match[1]:
                name = name2

            face_names.append(name)

        # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            # Draw a box around the face
            cv2.rectangle(frame, (left-22, top-42),
                          (right+22, bottom+32), (0, 0, 255), 2)

            crop_img = frame[top-40:bottom+30, left-20:right+20]
            if(name == name1):
                cv2.imwrite("./face_database/" + name1 + "/" + name1 + video_name.split('.')[0] +
                            str(count)+".png", crop_img)
                count = count + 1

            elif(name == name2):
                cv2.imwrite("./face_database/" + name2 + "/" + name2 + video_name.split('.')[0] +
                            str(count1)+".png", crop_img)
                count1 = count1 + 1
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left-10, bottom+10 - 35),
                          (right+10, bottom+10), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left-10 + 6, bottom+10 - 6),
                        font, 1.0, (255, 255, 255), 1)

        # Write the resulting image to the output video file

        print("Writing frame {} / {}".format(frame_number, length))

        cv2.imshow('face_recog_crop', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# All done!
input_movie.release()
cv2.destroyAllWindows()
