import os
import shutil
from random import shuffle
import math


source_dir = "./crater"
dest_dir1 = "images/train/Crater"
dest_dir2 = "images/valid/Crater"
os.makedirs(dest_dir1, exist_ok=True)
os.makedirs(dest_dir2, exist_ok=True)

files = [x for x in os.listdir(source_dir)]
shuffle(files)
idx80 = math.ceil(len(files) * 0.8)

for filename in files[:idx80]:
	source_path = os.path.join(source_dir, filename)
	shutil.copy(source_path, dest_dir1)

for filename in files[idx80: ]:
	source_path = os.path.join(source_dir, filename)
	shutil.copy(source_path, dest_dir2)


source_dir = "./nocrater"
dest_dir1 = "images/train/NoCrater"
dest_dir2 = "images/valid/NoCrater"
os.makedirs(dest_dir1, exist_ok=True)
os.makedirs(dest_dir2, exist_ok=True)

files = [x for x in os.listdir(source_dir)]
shuffle(files)
idx80 = math.ceil(len(files) * 0.8)

for filename in files[:idx80]:
	source_path = os.path.join(source_dir, filename)
	shutil.copy(source_path, dest_dir1)

for filename in files[idx80: ]:
	source_path = os.path.join(source_dir, filename)
	shutil.copy(source_path, dest_dir2)