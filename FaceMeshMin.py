import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Video/change_face_new.mp4")

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2) #Face object

mpDraw = mp.solutions.drawing_utils #Use this to draw default box of the detected face
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2) #Specify the draw function

pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #The default is BGR for video capture, convert to RGB as mediapipe using RGB
    results = faceMesh.process(imgRGB) #Detect the face object from the 

    if results.multi_face_landmarks: #If face lankmark is detected in the video
        h, w, c = img.shape #Get the video's width and height
        for landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, landmarks, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec) # Draw default facemesh dots
            for id, lm in enumerate(landmarks.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                print(id, x, y)
                # bboxC = detection.location_data.relative_bounding_box #Bounding box of the video
                # bbox = 
                # cv2.rectangle(img, bbox, (255,255,255),2)
                # cv2.putText(img, f'{int(detection.score[0] *100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2) #Produce the % of dection accuracy 

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),2)
    cv2.imshow("Image",img)
    cv2.waitKey(1)