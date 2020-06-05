import os.path
import time

import cv2
import torch
import torchvision.transforms as transforms

from detLib.models import Darknet
from detLib.utils.datasets import resize, pad_to_square
from detLib.utils.utils import rescale_boxes, non_max_suppression
from tools import clean_folder

color_list = [(222, 98, 61), (160, 90, 105), (140, 82, 148), (107, 75, 148), (76, 67, 149), (50, 113, 149),
              (23, 159, 149), (76, 180, 109), (131, 204, 72), (187, 197, 71), (242, 190, 69)]
le = len(color_list)


class Detector():
    def __init__(self, yolo_path="detLib/",
                 weights="weights/yolov3_ckpt_current_50.pth",
                 config="config/yolov3-custom.cfg",
                 output_dir="output",
                 resize_size=960, conf_thres=0.9, nms_thres=0.5,
                 verbose=False):
        """initaliase people detector"""
        
        self.yolo_path = yolo_path
        
        self.output_dir = output_dir
        
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        weights = "weights/det/yolov3_ckpt_current_50.pth"
        model_def = self.y_path("config/yolov3-custom.cfg")
        self.model = Darknet(model_def, img_size=416).to(self.device)
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.eval()
        self.classes = ['pedestrian']
        self.conf_thres, self.nms_thres = conf_thres, nms_thres
        self.resize_size = resize_size
    
    def y_path(self, p):
        """return join path in yolo folder"""
        return os.path.join(self.yolo_path, p)
    
    def detect(self, img):
        """detection on one image"""
        shape = img.shape[:2]
        img = transforms.ToTensor()(img)
        img, _ = pad_to_square(img, 0)
        
        if self.resize_size != 0:
            img = resize(img, self.resize_size)
        img.unsqueeze_(0)  # manually add batch axis
        img = img.to(self.device)
        with torch.no_grad():
            start_time = time.time()
            detections = self.model(img)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
            detections = detections[0]  # remove batch axis
            
            detections = rescale_boxes(detections, self.resize_size, shape)
        
        if self.verbose:
            print(f"--- %s seconds for {img.shape}---" % (time.time() - start_time))
        
        return detections
    
    def clip(self, taille, *args):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            l = []
            for x in args:
                x = min(max(x, 0.01), taille - 0.01)
                l.append(torch.tensor(x))
            return l
    
    def get_detections(self, img, detections):
        """get list of detected images"""
        det_list = []
        crop_list = []
        
        shape = img.shape
        
        for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
            (x1, x2), (y1, y2) = self.clip(shape[1], x1, x2), self.clip(shape[0], y1, y2)
            
            box_w = x2 - x1
            box_h = y2 - y1
            
            if 0.45 * box_h > box_w > 10:
                det_list.append([x1, y1, x2, y2])
                crop_list.append(img[int(y1):int(y2), int(x1):int(x2), :])
        
        return det_list, crop_list
    
    def save_detections_crop(self, img, detections, keep_folder=False):
        """save cropped peoples"""
        
        if not keep_folder:
            clean_folder(self.output_dir)
        
        start_time = time.time()
        
        for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
            box_w = x2 - x1
            box_h = y2 - y1
            if 0.45 * box_h > box_w > 10:
                pedestrian = img[int(y1):int(y2), int(x1):int(x2), :]
                filename = f'{self.output_dir}/{i:04}.png'
                cv2.imwrite(filename, pedestrian)
        
        if self.verbose:
            print(f"--- det %s seconds for saving {i} detections---" % (time.time() - start_time))
    
    def save_pic_with_detections(self, img, detections, ids=None, title="image"):
        idx = 0
        filename = f'{self.output_dir}/{title}.png'
        shape = img.shape
        
        for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
            (x1, x2), (y1, y2) = self.clip(shape[1], x1, x2), self.clip(shape[0], y1, y2)
            
            box_w = x2 - x1
            box_h = y2 - y1
            if 0.45 * box_h > box_w > 10:
                pedestrian = img[int(y1):int(y2), int(x1):int(x2), :]
                if ids != None:
                    col = color_list[ids[idx] % le]
                    cv2.putText(img, f'{ids[idx]}', (x1 + 1, y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, thickness=2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), col, thickness=2)
                    idx += 1
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 102, 255), 1)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img)
