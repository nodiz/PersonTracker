import argparse
import tqdm
import torch
import cv2
import sys
import time

from dr_utils import clean_folder, save_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videoname", type=str, default='vids/MOT16-10-raw.webm', help="videoname")
    parser.add_argument("--outputdir", type=str, default='output', help="video folder")
    parser.add_argument("--reiddir", type=str, default="", help="msmt-like detection folder")
    parser.add_argument("--workdir", type=str, default='temp', help="frames folder")
    parser.add_argument("--yolopath", type=str, default='detLib', help="outputdir")
    parser.add_argument("--reidpath", type=str, default='reidLib', help="outputdir")
    parser.add_argument("--threshold", type=float, default=1, help="threshold for new identity")

    opt = parser.parse_args()
    videoname = opt.videoname
    output_dir = opt.outputdir
    work_dir = opt.workdir
    reid_dir = opt.reiddir if opt.reiddir != "" else None
    clean_folder(output_dir)
    clean_folder(work_dir)
    clean_folder(reid_dir)

    sys.path.append(opt.yolopath)
    from detect3000 import Detector
    detector = Detector(yolo_path=opt.yolopath, output_dir=work_dir, verbose=False)
    from reid3000 import Reid
    reid = Reid(save_path=reid_dir, threshold=opt.threshold, verbose=False)
    
    videoframe = cv2.VideoCapture(videoname)
    framenr=0
    
    if not videoframe.isOpened() :
      print("Error opening video stream or file")
      
    while videoframe.isOpened():
      ret, img = videoframe.read()
      start = time.time()
      if ret:
        framenr+=1
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = detector.detect(img)
        det_list, crop_list = detector.get_detections(img, detections)
        det_ids = reid.evaluate_query(crop_list)
        detector.save_pic_with_detections(img, detections, det_ids, title=f"pic-{framenr:04}")
        print(f"Elaborating frame {framenr} freq: {1/(time.time() - start):.3}")
        
    videoframe.release()
    cv2.destroyAllWindows()

    save_video(work_dir, output_dir, filename="output.avi")
