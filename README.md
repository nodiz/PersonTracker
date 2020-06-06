# PersonTracker


## How to:

1. clone nodiz:this -> det-reid/

`git clone https://github.com/nodiz/YOLOv3-pedestrian.git det-reid/`

2. clone nodiz:yolo -> det-reid/detLib/

`git clone https://github.com/nodiz/YOLOv3-pedestrian.git det-reid/detlib/`

3. clone nodiz:reid -> det-reid/reidLib/

`git clone https://github.com/nodiz/reid-strong-baseline.git det-reid/reidLib/`

4. Copy weights -> det-reid/weights/

- weigths/reid/ : [backbone (resnet-ibn)](https://drive.google.com/file/d/1_r4wp14hEMkABVow58Xr4mPg7gvgOMto/view), [model](https://drive.google.com/drive/folders/1eq2Zpr2kn9FgAwpxDOl7eHaaHXa-m5lv?usp=sharing)
- weights/det/ : [model](https://drive.google.com/drive/folders/1DRPNNJoIbM7utW-kDCdCFr7m4gZ0BVvo?usp=sharing)

5. run pippo3000.py

## Example

```
python pippo3000.py --imgsize 512 --videoin MOT16-11-raw.webm --outname v2.avi --reiddir g2
```

Will process the video *vids/MOT16-11-raw.webm* and save output in *output/v2.avi*, eventually images will be converted to 512x512. 
During processing the gallery identifications will be saved in the folder *g2* (for every identity, a folder containing all the crops where it appears)

## Performance

You can run 5 fps on a 512x512 image size. Changes in size cause variation in the detection speed which is the bottleneck.
We are currently working to improve the speed.

## To-Do

A. Detector

• train better
• include more data augmentation (motion blur, scale)
• include ignore regions in the loss function
• train on ECP night dataset
• upgrade to yolo v4
• test other possibilities: RCNN, FCOS, DeTR...

B. Reidentification

• use the developed framework to gather even more data
• apply in more use cases (reid on phone pictures..)
• reduce identity matching complexity

C. Det+reid

• make faster
• tune the parameters better
• find some nice applications
