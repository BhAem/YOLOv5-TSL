import numpy as np
import cv2
import os
import time

video = "./videos/rain.mp4"
cap = cv2.VideoCapture(video)
FPS = cap.get(cv2.CAP_PROP_FPS)
print(FPS)

sta_min = [1,  1,  1,  2,  3]
sta_sec = [9,  23, 43, 29, 23]
end_min = [1,  1,  1,  2,  3]
end_sec = [12, 26, 47, 36, 25]
num = len(sta_min)

for i in range(num):

    START_HOUR = 0
    START_MIN = sta_min[i]
    START_SECOND = sta_sec[i]
    START_TIME = START_HOUR * 3600 + START_MIN * 60 + START_SECOND  # 设置开始时间(单位秒)
    END_HOUR = 0
    END_MIN = end_min[i]
    END_SECOND = end_sec[i]
    END_TIME = END_HOUR * 3600 + END_MIN * 60 + END_SECOND  # 设置结束时间(单位秒)
    size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(size[0])
    H = int(size[1])
    size = (W, H)
    TOTAL_FRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
    frameToStart = START_TIME * FPS  # 开始帧 = 开始时间*帧率
    print(frameToStart)
    frametoStop = END_TIME * FPS  # 结束帧 = 结束时间*帧率
    print(frametoStop)
    videoWriter =cv2.VideoWriter(f'./videos/rain{i}.avi',cv2.VideoWriter_fourcc('X','V','I','D'),FPS,size)

    # cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)  # 设置读取的位置,从第几帧开始读取视频
    COUNT = 0
    while True:
            success, frame = cap.read()
            if success:
                COUNT += 1
                if COUNT <= frametoStop and COUNT > frameToStart:  # 选取起始帧
                    print('correct= ', COUNT)
                    videoWriter.write(frame)
            # print('mistake= ', COUNT)
            if COUNT > frametoStop:
                break
    print(f'{i}, end')


import os
import cv2

# def save_image(image, address, num):
#     pic_address = address + str(num) + '.jpg'
#     cv2.imwrite(pic_address, image)
#
# ori = 'D:\\Users\\11939\\Desktop\\dark2\\'
# ori2 = 'D:\\Users\\11939\\Desktop\\dark2\\dark2-images\\'
# videos = os.listdir(ori)
# cnt = 0
# for video_name in videos:
#     video_path = os.path.join(ori, video_name)
#     videoCapture = cv2.VideoCapture(video_path)
#     frame_rate = int(videoCapture.get(cv2.CAP_PROP_FPS))
#     # frame_rate = 10
#     success, frame = videoCapture.read()
#     i = 0
#     while success:
#         i = i + 1
#         # 每隔固定帧保存一张图片
#         if i % frame_rate == 0:
#             cnt = cnt + 1
#             save_image(frame, ori2, cnt)
#             print(cnt)
#         success, frame = videoCapture.read()


# import os
# import cv2
# import random
# import shutil
#
# base = './real/snow/'
# img_test_path = 'images/test/'
# img_tran_path = 'images/train'
#
# lab_test_path = 'labels/test/'
# lab_tran_path = 'labels/train'
#
# imgs = os.listdir(os.path.join(base, img_test_path))
# random.shuffle(imgs)
# final = len(imgs) // 2
# print(final)
# cnt = 1
#
# for img in imgs:
#     print(cnt)
#     name = img.split('.')[0]
#     lab = name + '.txt'
#
#     ori_img_path = os.path.join(base, img_test_path, img)
#     dst_img_path = os.path.join(base, img_tran_path, img)
#     ori_lab_path = os.path.join(base, lab_test_path, lab)
#     dst_lab_path = os.path.join(base, lab_tran_path, lab)
#
#     shutil.move(ori_img_path, dst_img_path)
#     shutil.move(ori_lab_path, dst_lab_path)
#
#     if cnt == final:
#         break
#     cnt += 1


# import torch
#
# we = torch.load('./weights/best2.pt')
# we['epoch'] = -1
# we['best_fitness'] = None
# we['ema'] = None
# we['updates'] = None
# we['optimizer'] = None
# we['wandb_id'] = None
# torch.save(we, './weights/best2.pt')
# print(1)
#

# import shutil
# import os
#
# path1 = r"D:\Users\11939\Desktop\基于深度学习的复杂环境下交通信号标志检测\Complex\yolov5-6.0\runs\detect\exp8"
# path2 = r"D:\Users\11939\Desktop\dark2\dark2-images"
# path3 = r"D:\Users\11939\Desktop\dark22"
#
# imgs = os.listdir(path1)
# for name in imgs:
#     p1 = os.path.join(path2, name)
#     p2 = os.path.join(path3, name)
#     shutil.copy(p1, p2)
#     print(name)









