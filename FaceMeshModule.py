import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode = False, MaxFaces = 2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = minDetectionCon
        self.MaxFaces = MaxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.MaxFaces, self.minDetectionCon, self.minTrackCon)
        
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2) #Specify the draw function    
    
    def findFacesMesh(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #The default is BGR for video capture, convert to RGB as mediapipe using RGB
        self.results = self.faceMesh.process(imgRGB)
        h, w, c = img.shape
        faces = [] #capture the faces
        if self.results.multi_face_landmarks: #If face lankmark is detected in the video
            h, w, c = img.shape #Get the video's width and height
            for face_item in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, face_item, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec) # Draw default facemesh dots
                face = []
                for id, lm in enumerate(face_item.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    face.append([x,y])
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture("Video/change_face_new.mp4")
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFacesMesh(img)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),2)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()