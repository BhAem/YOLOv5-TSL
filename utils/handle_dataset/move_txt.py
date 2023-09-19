import os
import shutil

src_img_train = r"D:\Users\11939\Desktop\yolov5-6.0\新建文件夹\TrafficSignLight6-night\images\train"
src_img_test = r"D:\Users\11939\Desktop\yolov5-6.0\新建文件夹\TrafficSignLight6-night\images\test"

src_label_train_out = r"D:\Users\11939\Desktop\yolov5-6.0\新建文件夹\TrafficSignLight6-night\labels\train-out"
dst_label_train = r"D:\Users\11939\Desktop\yolov5-6.0\新建文件夹\TrafficSignLight6-night\labels\train"
dst_label_test = r"D:\Users\11939\Desktop\yolov5-6.0\新建文件夹\TrafficSignLight6-night\labels\test"

txt_train_out = [s.split(".")[0] for s in os.listdir(src_label_train_out)]
img_trains = [s.split(".")[0] for s in os.listdir(src_img_train)]
img_tests = [s.split(".")[0] for s in os.listdir(src_img_test)]

for file in txt_train_out:
    name = file + ".txt"
    if file in img_trains:
        src_path = os.path.join(src_label_train_out, name)
        dst_path = os.path.join(dst_label_train, name)
        shutil.move(src_path, dst_path)
        print(file)
    elif file in img_tests:
        src_path = os.path.join(src_label_train_out, name)
        dst_path = os.path.join(dst_label_test, name)
        shutil.move(src_path, dst_path)
        print(file)
