import argparse
import os.path
import sys
import time

import cv2

from tools import clean_folder, clear_folder, make_folder, save_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # interesting to change
    parser.add_argument("--videoin", type=str, default='MOT16-10-raw.webm', help="input video (in vids/ folder)")
    parser.add_argument("--threshold", type=float, default=3, help="threshold for adding new identity")
    parser.add_argument("--imgsize", type=int, default=960, help="imagesize for detection")
    parser.add_argument("--detconf", type=float, default=0.9, help="required condifence for detection")
    # usually default
    parser.add_argument("--indir", type=str, default='vids', help="input video directory")
    parser.add_argument("--outname", type=str, default='output.avi', help="name of output video")
    parser.add_argument("--outdir", type=str, default='output', help="output video folder")
    parser.add_argument("--reiddir", type=str, default="", help="msmt-like detection folder")
    parser.add_argument("--workdir", type=str, default='temp', help="frames folder")
    parser.add_argument("--yolopath", type=str, default='detLib', help="outputdir")
    parser.add_argument("--reidpath", type=str, default='reidLib', help="outputdir")

    opt = parser.parse_args()
    
    videoname = os.path.join(opt.indir,opt.videoin)
    output_dir = opt.outdir
    work_dir = opt.workdir
    reid_dir = opt.reiddir if opt.reiddir != "" else None
    make_folder(output_dir)
    clean_folder(work_dir)
    clean_folder(reid_dir)
    
    sys.path.append(opt.yolopath)
    from detect import Detector
    
    detector = Detector(conf_thres=opt.detconf, yolo_path=opt.yolopath, resize_size=opt.imgsize, output_dir=work_dir)
    from reid import Reid
    
    reid = Reid(save_path=reid_dir, threshold=opt.threshold, verbose=False)
    
    videoframe = cv2.VideoCapture(videoname)
    framenr = 0
    
    if not videoframe.isOpened():
        print("Error opening video stream or file")
    
    while videoframe.isOpened():
        ret, img = videoframe.read()
        
        if not ret:  # reached end of video
            break
        
        start = time.time()
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = detector.detect(img)
        det_list, crop_list = detector.get_detections(img, detections)
        det_ids = reid.evaluate_query(crop_list)
        detector.save_pic_with_detections(img, detections, det_ids, title=f"pic-{framenr:04}")
        
        print(f"Elaborating frame {framenr} fps: {1 / (time.time() - start):.3}")
        
        framenr += 1
    
    print("Processing finished, saving video")
    videoframe.release()
    cv2.destroyAllWindows()
    
    save_video(work_dir, output_dir, filename=opt.outname)
    
    print(f"Video saved as {os.path.join(output_dir, opt.outname)}")
    clear_folder(work_dir)
