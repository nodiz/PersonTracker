import glob
import os.path
import shutil
import argparse

import cv2


def clean_folder(folder_name):
    try:
        shutil.rmtree(folder_name)
    except OSError:
        pass
    except TypeError:
        return
    os.makedirs(folder_name, exist_ok=True)


def make_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)


def clear_folder(folder_name):
    try:
        shutil.rmtree(folder_name)
    except OSError as e:
        pass
    except TypeError:
        return


def save_video(folder_in, folder_out, filename="video"):
    file_array = [filename for filename in glob.glob(os.path.join(folder_in, '*.png'))]
    file_array.sort()
    img_array = [cv2.imread(f) for f in file_array]
    
    img = img_array[0]
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    
    out = cv2.VideoWriter(os.path.join(folder_out, filename), cv2.VideoWriter_fourcc(*'DIVX'), 7, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, default='video', help="videoname")
    opt = parser.parse_args()
    
    if opt.func == "video":
        save_video("temp", "output", "ouput.avi")
    elif opt.func == "clean":
        clean_folder("temp")
        clean_folder("output")
        clean_folder("pippo")
