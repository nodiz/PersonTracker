import sys, os.path, shutil, time
import torch
import cv2
import torchvision.transforms as transforms
import sys

from detLib.models import Darknet
from detLib.utils.datasets import resize, pad_to_square
from detLib.utils.utils import rescale_boxes, non_max_suppression

from dr_utils import clean_folder


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
        
        weights = self.y_path("weights/yolov3_ckpt_current_50.pth")
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
        
        with torch.no_grad():
            start_time = time.time()
            detections = self.model(img)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
            detections = detections[0]  # remove batch axis
    
            detections = rescale_boxes(detections, self.resize_size, shape)
      
        if self.verbose:
            print(f"--- %s seconds for {img.shape}---" % (time.time() - start_time))

        return detections
    
    def get_detections(self, img, detections):
        """get list of detected images"""
        
        crop_list = []
        
        for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
            box_w = x2 - x1
            box_h = y2 - y1
        
            if 0.45 * box_h > box_w > 10:
                crop_list.append(img[int(y1):int(y2), int(x1):int(x2), :])
            
            return crop_list

    def save_detections(self, img, detections, keep_folder=False):
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
            print(f"--- %s seconds for saving {i} detections---" % (time.time() - start_time))
        