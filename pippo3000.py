import argparse
import tqdm
import torch
import cv2
import sys

from dr_utils import clean_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videoname", type=str, default='VideoToTrack/MOT16-10-raw.webm', help="videoname")
    parser.add_argument("--outputdir", type=str, default='output', help="outputdir")
    parser.add_argument("--workdir", type=str, default='temp', help="outputdir")
    parser.add_argument("--yolopath", type=str, default='detLib', help="outputdir")
    parser.add_argument("--reidpath", type=str, default='reidLib', help="outputdir")

    opt = parser.parse_args()
    videoname = opt.videoname
    output_dir = opt.outputdir
    work_dir = opt.workdir
    clean_folder(output_dir)
    clean_folder(work_dir)

    sys.path.append(opt.yolopath)
    from detect3000 import Detector
    detector = Detector(yolo_path=opt.yolopath, output_dir=work_dir, verbose=True)
    from reid3000 import Reid
    reid = Reid()
    
    videoframe = cv2.VideoCapture(videoname)
    framenr=0
    
    if not videoframe.isOpened() :
      print("Error opening video stream or file")
      
    while videoframe.isOpened():
      ret, img = videoframe.read()
      if ret:
        framenr+=1
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = detector.detect(img)
        detector.save_detections(img, detections)
        # reid.evaluate_query(work_dir)  # implementation not finished
        
        break
    
    videoframe.release()
    cv2.destroyAllWindows()