import shutil
import os
import random

folder = "TrafficSignLight9-daytime"
folder1 = "TrafficSignLight9-fog"
folder2 = "TrafficSignLight9-rainsnow"
folder3 = "TrafficSignLight9-night"

src_img_train = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder}\images\train"
src_lab_train = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder}\labels\train"
dst_img_train = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder}\images\train"
dst_lab_train = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder}\labels\train"

src_img_train1 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder1}\images\train"
src_lab_train1 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder1}\labels\train"
dst_img_train1 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder1}\images\train"
dst_lab_train1 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder1}\labels\train"

src_img_train2 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder2}\images\train"
src_lab_train2 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder2}\labels\train"
dst_img_train2 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder2}\images\train"
dst_lab_train2 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder2}\labels\train"

src_img_train3 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder3}\images\train"
src_lab_train3 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder3}\labels\train"
dst_img_train3 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder3}\images\train"
dst_lab_train3 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder3}\labels\train"

imgs = [s.split('.')[0] for s in os.listdir(src_img_train)]
current_data_length = len(imgs)
current_data_index_list = list(range(current_data_length))
random.shuffle(current_data_index_list)
train_stop_flag = 500
current_idx = 0

for i in current_data_index_list:
    src_img = os.path.join(src_img_train, imgs[i] + '.jpg')
    dst_img = os.path.join(dst_img_train, imgs[i] + '.jpg')
    src_lb = os.path.join(src_lab_train, imgs[i] + '.txt')
    dst_lb = os.path.join(dst_lab_train, imgs[i] + '.txt')
    shutil.copy(src_img, dst_img)
    shutil.copy(src_lb, dst_lb)

    src_img = os.path.join(src_img_train1, imgs[i] + '_fog.jpg')
    dst_img = os.path.join(dst_img_train1, imgs[i] + '_fog.jpg')
    src_lb = os.path.join(src_lab_train1, imgs[i] + '_fog.txt')
    dst_lb = os.path.join(dst_lab_train1, imgs[i] + '_fog.txt')
    shutil.copy(src_img, dst_img)
    shutil.copy(src_lb, dst_lb)

    src_img = os.path.join(src_img_train2, imgs[i] + '_rainsnow.jpg')
    dst_img = os.path.join(dst_img_train2, imgs[i] + '_rainsnow.jpg')
    src_lb = os.path.join(src_lab_train2, imgs[i] + '_rainsnow.txt')
    dst_lb = os.path.join(dst_lab_train2, imgs[i] + '_rainsnow.txt')
    shutil.copy(src_img, dst_img)
    shutil.copy(src_lb, dst_lb)

    src_img = os.path.join(src_img_train3, imgs[i] + '_night.jpg')
    dst_img = os.path.join(dst_img_train3, imgs[i] + '_night.jpg')
    src_lb = os.path.join(src_lab_train3, imgs[i] + '_night.txt')
    dst_lb = os.path.join(dst_lab_train3, imgs[i] + '_night.txt')
    shutil.copy(src_img, dst_img)
    shutil.copy(src_lb, dst_lb)

    print(current_idx)
    current_idx = current_idx + 1
    if current_idx == train_stop_flag:
        break


src_img_train = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder}\images\test"
src_lab_train = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder}\labels\test"
dst_img_train = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder}\images\test"
dst_lab_train = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder}\labels\test"

src_img_train1 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder1}\images\test"
src_lab_train1 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder1}\labels\test"
dst_img_train1 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder1}\images\test"
dst_lab_train1 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder1}\labels\test"

src_img_train2 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder2}\images\test"
src_lab_train2 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder2}\labels\test"
dst_img_train2 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder2}\images\test"
dst_lab_train2 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder2}\labels\test"

src_img_train3 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder3}\images\test"
src_lab_train3 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\{folder3}\labels\test"
dst_img_train3 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder3}\images\test"
dst_lab_train3 = rf"D:\Users\11939\Desktop\Complex\yolov5-6.0\000\lzr\{folder3}\labels\test"

imgs = [s.split('.')[0] for s in os.listdir(src_img_train)]
current_data_length = len(imgs)
current_data_index_list = list(range(current_data_length))
random.shuffle(current_data_index_list)
train_stop_flag = 125
current_idx = 0

for i in current_data_index_list:
    src_img = os.path.join(src_img_train, imgs[i] + '.jpg')
    dst_img = os.path.join(dst_img_train, imgs[i] + '.jpg')
    src_lb = os.path.join(src_lab_train, imgs[i] + '.txt')
    dst_lb = os.path.join(dst_lab_train, imgs[i] + '.txt')
    shutil.copy(src_img, dst_img)
    shutil.copy(src_lb, dst_lb)

    src_img = os.path.join(src_img_train1, imgs[i] + '_fog.jpg')
    dst_img = os.path.join(dst_img_train1, imgs[i] + '_fog.jpg')
    src_lb = os.path.join(src_lab_train1, imgs[i] + '_fog.txt')
    dst_lb = os.path.join(dst_lab_train1, imgs[i] + '_fog.txt')
    shutil.copy(src_img, dst_img)
    shutil.copy(src_lb, dst_lb)

    src_img = os.path.join(src_img_train2, imgs[i] + '_rainsnow.jpg')
    dst_img = os.path.join(dst_img_train2, imgs[i] + '_rainsnow.jpg')
    src_lb = os.path.join(src_lab_train2, imgs[i] + '_rainsnow.txt')
    dst_lb = os.path.join(dst_lab_train2, imgs[i] + '_rainsnow.txt')
    shutil.copy(src_img, dst_img)
    shutil.copy(src_lb, dst_lb)

    src_img = os.path.join(src_img_train3, imgs[i] + '_night.jpg')
    dst_img = os.path.join(dst_img_train3, imgs[i] + '_night.jpg')
    src_lb = os.path.join(src_lab_train3, imgs[i] + '_night.txt')
    dst_lb = os.path.join(dst_lab_train3, imgs[i] + '_night.txt')
    shutil.copy(src_img, dst_img)
    shutil.copy(src_lb, dst_lb)

    print(current_idx)
    current_idx = current_idx + 1
    if current_idx == train_stop_flag:
        break


