import os
import shutil

src_file = r"D:\Users\11939\Desktop\yolov5-6.0\TrafficSignLight6-rainsnow\labels\train"
dst_file = r"D:\Users\11939\Desktop\yolov5-6.0\TrafficSignLight6-rainsnow\labels\train-rename"

files = os.listdir(src_file)

for file in files:
    name = file.split(".")[0]
    suffix = file.split(".")[1]
    new_name = name + "_rainsnow." + suffix
    src_path = os.path.join(src_file, file)
    dst_path = os.path.join(dst_file, new_name)
    shutil.move(src_path, dst_path)
    print(file)

# for file in files:
#     name = file.split(".")[0]
#     o_name = name[:-4]
#     suffix = file.split(".")[1]
#     new_name = o_name + "." + suffix
#     src_path = os.path.join(src_file, file)
#     dst_path = os.path.join(dst_file, new_name)
#     shutil.move(src_path, dst_path)
#     print(file)