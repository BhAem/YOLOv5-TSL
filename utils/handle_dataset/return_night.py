import os
import shutil

src_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8\images\train"
src_lab_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8\labels\train"

dst_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-night\images\train"
dst_lab_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-night\labels\train"

imgs = [s.split(".")[0] for s in os.listdir(src_img_path)]

for img in imgs:
    if img.__contains__("_night"):
        src_img = os.path.join(src_img_path, img+".jpg")
        dst_img = os.path.join(dst_img_path, img+".jpg")

        src_lab = os.path.join(src_lab_path, img+".txt")
        dst_lab = os.path.join(dst_lab_path, img+".txt")

        shutil.copy(src_img, dst_img)
        shutil.copy(src_lab, dst_lab)
        print(img)