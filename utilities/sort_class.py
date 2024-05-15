import os
import shutil
from PIL import Image, ImageDraw

def copy_files_by_name(source_dir, dest_dir1, dest_dir2, str1, str2):
    # Create destination directories if they don't exist
    os.makedirs(dest_dir1, exist_ok=True)
    os.makedirs(dest_dir2, exist_ok=True)
	
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        if not os.path.isfile(source_path):
             continue
        if Image.open(source_path).size != (224,224):
             print(source_path, " not 224,224!")
             continue
        if str1 in filename:
            shutil.copy(source_path, dest_dir1)
        elif str2 in filename:
            shutil.copy(source_path, dest_dir2)

# Example usage:
source_dir = "./Batch 3/"
dest_dir1 = "crater/"
dest_dir2 = "nocrater/"
str1 = "_crater_"
str2 = "_nocrater_"

copy_files_by_name(source_dir, dest_dir1, dest_dir2, str1, str2)