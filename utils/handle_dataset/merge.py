import os
import shutil

src_file = r"D:\Users\11939\Desktop\yolov5-6.0\TrafficSignLight6-rainsnow\images\train"
dst_file = r"D:\Users\11939\Desktop\yolov5-6.0\TrafficSignLight6\images\train"

files = os.listdir(src_file)

for file in files:
    src_path = os.path.join(src_file, file)
    dst_path = os.path.join(dst_file, file)
    shutil.copy(src_path, dst_path)
    print(file)