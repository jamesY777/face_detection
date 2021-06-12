import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(self.minDetectionCon) #Face object
        self.mpDraw = mp.solutions.drawing_utils #Use this to draw default box of the detected face
    
    def findFaces(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #The default is BGR for video capture, convert to RGB as mediapipe using RGB
        self.results = self.face.process(imgRGB)
        bboxs = []
        h, w, c = img.shape
        if self.results.detections: #If face is detected in the video
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box #Bounding box of the video
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] *100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2) #Produce the % of dection accuracy
                bboxs.append([id, bbox, detection.score])
        return img, bboxs
    
    # A little more highlighted box for some fancy draw
    def fancyDraw(self, img, bbox, l=15, t=3, rt=1):
        x, y, w, h = bbox #x y is the left top corner point
        x1, y1 = x+w, y+h #x1, y1 is the right bottom corner point

        #Top left corner
        cv2.line(img, (x,y), (x+l, y),(255,255,255), t) #(img, starting point)
        cv2.line(img, (x,y), (x, y+l),(255,255,255), t)

        #Top right corner
        cv2.line(img, (x1,y), (x1-l, y),(255,255,255), t) #(img, starting point)
        cv2.line(img, (x1,y), (x1, y+l),(255,255,255), t)

        #Bottom left
        cv2.line(img, (x,y1), (x+l, y1),(255,255,255), t) #(img, starting point)
        cv2.line(img, (x,y1), (x, y1-l),(255,255,255), t) 

        #Bottom right
        cv2.line(img, (x1,y1), (x1-l, y1),(255,255,255), t) #(img, starting point)
        cv2.line(img, (x1,y1), (x1, y1-l),(255,255,255), t) 
        cv2.rectangle(img, bbox, (255,255,255), rt)
        return img
def main():
    cap = cv2.VideoCapture("Video/change_face_new.mp4")
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img, draw=False)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,25),2)
        cv2.imshow("Image",img)
        cv2.waitKey(20)

if __name__ == "__main__":
    main()