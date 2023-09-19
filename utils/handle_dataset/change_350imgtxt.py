import random
import os
import shutil

# train_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-daytime\images\train"
# test_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-daytime\images\test"
# train_fog_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-fog\images\train"
# test_fog_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-fog\images\test"
train_night_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-night\images\train"
test_night_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-night\images\test"
# train_rn_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-rainsnow\images\train"
# test_rn_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-rainsnow\images\test"

# train2test_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-daytime\images\train2test"
# test2train_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-daytime\images\test2train"
# train2test_fog_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-fog\images\train2test"
# test2train_fog_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-fog\images\test2train"
train2test_night_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-night\images\train2test"
test2train_night_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-night\images\test2train"
# train2test_rn_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-rainsnow\images\train2test"
# test2train_rn_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-rainsnow\images\test2train"

# train_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-daytime\labels\train"
# test_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-daytime\labels\test"
# train_fog_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-fog\labels\train"
# test_fog_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-fog\labels\test"
train_night_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-night\labels\train"
test_night_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-night\labels\test"
# train_rn_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-rainsnow\labels\train"
# test_rn_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-rainsnow\labels\test"

# train2test_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-daytime\labels\train2test"
# test2train_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-daytime\labels\test2train"
# train2test_fog_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-fog\labels\train2test"
# test2train_fog_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-fog\labels\test2train"
train2test_night_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-night\labels\train2test"
test2train_night_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-night\labels\test2train"
# train2test_rn_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-rainsnow\labels\train2test"
# test2train_rn_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\000\TrafficSignLight8-rainsnow\labels\test2train"


imgs = [s.split(".")[0] for s in os.listdir(train_night_img_path)]
current_data_length = len(imgs)
current_data_index_list = list(range(current_data_length))
random.shuffle(current_data_index_list)

cnt = 0
for i in current_data_index_list:
    # src_img_path = os.path.join(train_img_path, imgs[i]+".jpg")
    # dst_img_path = os.path.join(train2test_img_path, imgs[i]+".jpg")
    # src_txt_path = os.path.join(train_txt_path, imgs[i]+".txt")
    # dst_txt_path = os.path.join(train2test_txt_path, imgs[i]+".txt")
    # shutil.move(src_img_path, dst_img_path)
    # shutil.move(src_txt_path, dst_txt_path)
    #
    # src_img_path = os.path.join(train_fog_img_path, imgs[i]+"_fog.jpg")
    # dst_img_path = os.path.join(train2test_fog_img_path, imgs[i]+"_fog.jpg")
    # src_txt_path = os.path.join(train_fog_txt_path, imgs[i]+"_fog.txt")
    # dst_txt_path = os.path.join(train2test_fog_txt_path, imgs[i]+"_fog.txt")
    # shutil.move(src_img_path, dst_img_path)
    # shutil.move(src_txt_path, dst_txt_path)
    #
    src_img_path = os.path.join(train_night_img_path, imgs[i]+".jpg")
    dst_img_path = os.path.join(train2test_night_img_path, imgs[i]+".jpg")
    src_txt_path = os.path.join(train_night_txt_path, imgs[i]+".txt")
    dst_txt_path = os.path.join(train2test_night_txt_path, imgs[i]+".txt")
    shutil.move(src_img_path, dst_img_path)
    shutil.move(src_txt_path, dst_txt_path)
    #
    # src_img_path = os.path.join(train_rn_img_path, imgs[i]+"_rainsnow.jpg")
    # dst_img_path = os.path.join(train2test_rn_img_path, imgs[i]+"_rainsnow.jpg")
    # src_txt_path = os.path.join(train_rn_txt_path, imgs[i]+"_rainsnow.txt")
    # dst_txt_path = os.path.join(train2test_rn_txt_path, imgs[i]+"_rainsnow.txt")
    # shutil.move(src_img_path, dst_img_path)
    # shutil.move(src_txt_path, dst_txt_path)

    print(cnt)
    cnt += 1
    if cnt == 50:
        break


imgs = [s.split(".")[0] for s in os.listdir(test_night_img_path)]
current_data_length = len(imgs)
current_data_index_list = list(range(current_data_length))
random.shuffle(current_data_index_list)

cnt = 0
for i in current_data_index_list:
    # src_img_path = os.path.join(test_img_path, imgs[i]+".jpg")
    # dst_img_path = os.path.join(test2train_img_path, imgs[i]+".jpg")
    # src_txt_path = os.path.join(test_txt_path, imgs[i]+".txt")
    # dst_txt_path = os.path.join(test2train_txt_path, imgs[i]+".txt")
    # shutil.move(src_img_path, dst_img_path)
    # shutil.move(src_txt_path, dst_txt_path)
    #
    # src_img_path = os.path.join(test_fog_img_path, imgs[i]+"_fog.jpg")
    # dst_img_path = os.path.join(test2train_fog_img_path, imgs[i]+"_fog.jpg")
    # src_txt_path = os.path.join(test_fog_txt_path, imgs[i]+"_fog.txt")
    # dst_txt_path = os.path.join(test2train_fog_txt_path, imgs[i]+"_fog.txt")
    # shutil.move(src_img_path, dst_img_path)
    # shutil.move(src_txt_path, dst_txt_path)
    #
    src_img_path = os.path.join(test_night_img_path, imgs[i]+".jpg")
    dst_img_path = os.path.join(test2train_night_img_path, imgs[i]+".jpg")
    src_txt_path = os.path.join(test_night_txt_path, imgs[i]+".txt")
    dst_txt_path = os.path.join(test2train_night_txt_path, imgs[i]+".txt")
    shutil.move(src_img_path, dst_img_path)
    shutil.move(src_txt_path, dst_txt_path)
    #
    # src_img_path = os.path.join(test_rn_img_path, imgs[i]+"_rainsnow.jpg")
    # dst_img_path = os.path.join(test2train_rn_img_path, imgs[i]+"_rainsnow.jpg")
    # src_txt_path = os.path.join(test_rn_txt_path, imgs[i]+"_rainsnow.txt")
    # dst_txt_path = os.path.join(test2train_rn_txt_path, imgs[i]+"_rainsnow.txt")
    # shutil.move(src_img_path, dst_img_path)
    # shutil.move(src_txt_path, dst_txt_path)

    print(cnt)
    cnt += 1
    if cnt == 50:
        break