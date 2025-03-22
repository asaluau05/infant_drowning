# camera.py
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import time
import numpy as np
import sklearn
import joblib
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import albumentations
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import argparse

import PIL.Image
from PIL import Image

lb = joblib.load('lb.pkl')
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # changed 3 to 1
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, len(lb.classes_))
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
            
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        ff=open("file.txt","r")
        video=ff.read()
        ff.close()
        
        self.video = cv2.VideoCapture(video)
        self.k=1
        ########
        #slef.lb = joblib.load('lb.pkl')
        


        print('Loading model and label binarizer...')
        self.lb = joblib.load('lb.pkl')
        self.model = CustomCNN()
        print('Model Loaded...')
        self.model.load_state_dict(torch.load('model.pth', map_location='cpu'))
        self.model.eval()
        print('Loaded model state_dict...')
        self.aug = albumentations.Compose([
            albumentations.Resize(224, 224),
            ])

        self.t0 = time.time() #gives time in seconds after 1970

        #def detectDrowning(source):
        self.isDrowning = False
        self.fram=0
        #####
    
    def __del__(self):
        self.video.release()
        
    
    def get_frame(self):
        status, frame = self.video.read()
        #self.out.write(image)
        #status, frame = cap.read()

        # apply object detection
        bbox, label, conf = cv.detect_common_objects(frame)
        
        # if only one person is detected, use model-based detection
        if len(bbox) == 1:
            bbox0 = bbox[0]
            #centre = np.zeros(s)
            centre = [0,0]


            for i in range(0, len(bbox)):
                centre[i] =[(bbox[i][0]+bbox[i][2])/2,(bbox[i][1]+bbox[i][3])/2 ]

            centre =[(bbox0[0]+bbox0[2])/2,(bbox0[1]+bbox0[3])/2 ]
            
            start_time = time.time()
            self.model.eval()
            with torch.no_grad():
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_image = self.aug(image=np.array(pil_image))['image']
                if self.fram == 500:
                    s=1
                    
                    
                #break
                self.fram+=1
                pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
                pil_image = torch.tensor(pil_image, dtype=torch.float).cpu()
                pil_image = pil_image.unsqueeze(0)
                outputs = self.model(pil_image)
                _, preds = torch.max(outputs.data, 1)

            print("Swimming status : ",self.lb.classes_[preds])
            if(self.lb.classes_[preds] =='drowning'):
                self.isDrowning = True
                ff=open("result.txt","w")
                ff.write("Drowning")
                ff.close()
            if(self.lb.classes_[preds] =='normal'):
                ff=open("result.txt","w")
                ff.write("Normal")
                ff.close()
                self.isDrowning = False
            out = draw_bbox(frame, bbox, label, conf,self.isDrowning)
            
        # if more than one person is detected, use logic-based detection
        elif len(bbox) > 1:
            # calculate the centroid of each bounding box
            centres = []
            for i in range(len(bbox)):
                bbox_i = bbox[i]
                centre_i = [(bbox_i[0] + bbox_i[2])/2, (bbox_i[1] + bbox_i[3])/2]
                centres.append(centre_i)
            
            # calculate the distance between each pair of centroids
            distances = []
            for i in range(len(centres)):
                for j in range(i+1, len(centres)):
                    dist = np.sqrt((centres[i][0] - centres[j][0])**2 + (centres[i][1] - centres[j][1])**2)
                    distances.append(dist)
            
            # if the minimum distance is less than a threshold, consider it as drowning
            if len(distances) > 0 and min(distances) < 50:
                self.isDrowning = True
            else:
                self.isDrowning = False
            out = draw_bbox(frame, bbox, label, conf, isDrowning)

        else:
            out = frame
        

            
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
