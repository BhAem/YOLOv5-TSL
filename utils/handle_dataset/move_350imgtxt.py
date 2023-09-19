import random
import os
import shutil

test_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-daytime\images\test"
test_fog_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-fog\images\test"
# test_night_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-night\images\test"
test_rn_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-rainsnow\images\test"
#
test2train_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-daytime\images\350"
test2train_fog_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-fog\images\350"
# test2train_night_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-night\images\350"
test2train_rn_img_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-rainsnow\images\350"
#
test_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-daytime\labels\test"
test_fog_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-fog\labels\test"
# test_night_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-night\labels\test"
test_rn_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-rainsnow\labels\test"
#
test2train_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-daytime\labels\350"
test2train_fog_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-fog\labels\350"
# test2train_night_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-night\labels\350"
test2train_rn_txt_path = r"D:\Users\11939\Desktop\yolov5-6.0\111\TrafficSignLight8-rainsnow\labels\350"

imgs = [s.split(".")[0] for s in os.listdir(test_img_path)]
current_data_length = len(imgs)
current_data_index_list = list(range(current_data_length))
random.shuffle(current_data_index_list)

cnt = 0
for i in current_data_index_list:
    src_img_path = os.path.join(test_img_path, imgs[i]+".jpg")
    dst_img_path = os.path.join(test2train_img_path, imgs[i]+".jpg")
    src_txt_path = os.path.join(test_txt_path, imgs[i]+".txt")
    dst_txt_path = os.path.join(test2train_txt_path, imgs[i]+".txt")
    shutil.move(src_img_path, dst_img_path)
    shutil.move(src_txt_path, dst_txt_path)
    #
    src_img_path = os.path.join(test_fog_img_path, imgs[i]+"_fog.jpg")
    dst_img_path = os.path.join(test2train_fog_img_path, imgs[i]+"_fog.jpg")
    src_txt_path = os.path.join(test_fog_txt_path, imgs[i]+"_fog.txt")
    dst_txt_path = os.path.join(test2train_fog_txt_path, imgs[i]+"_fog.txt")
    shutil.move(src_img_path, dst_img_path)
    shutil.move(src_txt_path, dst_txt_path)
    #
    # src_img_path = os.path.join(test_night_img_path, imgs[i]+".jpg")
    # dst_img_path = os.path.join(test2train_night_img_path, imgs[i]+".jpg")
    # src_txt_path = os.path.join(test_night_txt_path, imgs[i]+".txt")
    # dst_txt_path = os.path.join(test2train_night_txt_path, imgs[i]+".txt")
    # shutil.move(src_img_path, dst_img_path)
    # shutil.move(src_txt_path, dst_txt_path)
    #
    src_img_path = os.path.join(test_rn_img_path, imgs[i]+"_rainsnow.jpg")
    dst_img_path = os.path.join(test2train_rn_img_path, imgs[i]+"_rainsnow.jpg")
    src_txt_path = os.path.join(test_rn_txt_path, imgs[i]+"_rainsnow.txt")
    dst_txt_path = os.path.join(test2train_rn_txt_path, imgs[i]+"_rainsnow.txt")
    shutil.move(src_img_path, dst_img_path)
    shutil.move(src_txt_path, dst_txt_path)

    print(cnt)
    cnt += 1
    if cnt == 350:
        break