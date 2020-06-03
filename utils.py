import sys, os.path, shutil


def clean_folder(folder_name):
    try:
        shutil.rmtree(folder_name)
    except OSError as e:
        pass
    os.makedirs(folder_name, exist_ok=True)
