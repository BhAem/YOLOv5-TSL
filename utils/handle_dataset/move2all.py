import os
import shutil

dst_img_train_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8\images\train"
dst_img_test_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8\images\test"
dst_lab_train_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8\labels\train"
dst_lab_test_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8\labels\test"

src_daytime_img_train_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-daytime\images\train"
src_daytime_img_test_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-daytime\images\test"
src_daytime_lab_train_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-daytime\labels\train"
src_daytime_lab_test_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-daytime\labels\test"

src_fog_img_train_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-fog\images\train"
src_fog_img_test_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-fog\images\test"
src_fog_lab_train_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-fog\labels\train"
src_fog_lab_test_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-fog\labels\test"

src_night_img_train_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-night\images\train"
src_night_img_test_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-night\images\test"
src_night_lab_train_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-night\labels\train"
src_night_lab_test_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-night\labels\test"

src_rainsnow_img_train_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-rainsnow\images\train"
src_rainsnow_img_test_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-rainsnow\images\test"
src_rainsnow_lab_train_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-rainsnow\labels\train"
src_rainsnow_lab_test_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-rainsnow\labels\test"


imgs = [s.split(".")[0] for s in os.listdir(src_night_img_test_path)]
current_data_length = len(imgs)
current_data_index_list = list(range(current_data_length))

cnt = 0
for i in current_data_index_list:
    # src_img_path = os.path.join(src_daytime_img_test_path, imgs[i]+".jpg")
    # dst_img_path = os.path.join(dst_img_test_path, imgs[i]+".jpg")
    # src_txt_path = os.path.join(src_daytime_lab_test_path, imgs[i]+".txt")
    # dst_txt_path = os.path.join(dst_lab_test_path, imgs[i]+".txt")
    # shutil.copy(src_img_path, dst_img_path)
    # shutil.copy(src_txt_path, dst_txt_path)
    #
    # src_img_path = os.path.join(src_fog_img_test_path, imgs[i]+"_fog.jpg")
    # dst_img_path = os.path.join(dst_img_test_path, imgs[i]+"_fog.jpg")
    # src_txt_path = os.path.join(src_fog_lab_test_path, imgs[i]+"_fog.txt")
    # dst_txt_path = os.path.join(dst_lab_test_path, imgs[i]+"_fog.txt")
    # shutil.copy(src_img_path, dst_img_path)
    # shutil.copy(src_txt_path, dst_txt_path)
    #
    src_img_path = os.path.join(src_night_img_test_path, imgs[i]+".jpg")
    dst_img_path = os.path.join(dst_img_test_path, imgs[i]+".jpg")
    src_txt_path = os.path.join(src_night_lab_test_path, imgs[i]+".txt")
    dst_txt_path = os.path.join(dst_lab_test_path, imgs[i]+".txt")
    shutil.copy(src_img_path, dst_img_path)
    shutil.copy(src_txt_path, dst_txt_path)
    #
    # src_img_path = os.path.join(src_rainsnow_img_test_path, imgs[i]+"_rainsnow.jpg")
    # dst_img_path = os.path.join(dst_img_test_path, imgs[i]+"_rainsnow.jpg")
    # src_txt_path = os.path.join(src_rainsnow_lab_test_path, imgs[i]+"_rainsnow.txt")
    # dst_txt_path = os.path.join(dst_lab_test_path, imgs[i]+"_rainsnow.txt")
    # shutil.copy(src_img_path, dst_img_path)
    # shutil.copy(src_txt_path, dst_txt_path)

    print(cnt)
    cnt += 1
