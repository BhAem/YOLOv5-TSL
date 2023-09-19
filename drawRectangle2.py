import os
import cv2
import torch
import numpy as np

'''
    Normal: Faster    x1, y1, x11, y11 = 656, 294, 679, 316
            DETR      x1, y1, x11, y11 = 651, 291, 675, 314
            YOLOv6    x1, y1, x11, y11 = 649, 293, 672, 312
            MTSD      x1, y1, x11, y11 = 649, 293, 674, 313   x1, y1, x11, y11 = 624, 364, 645, 376
            x1, y1, x2, y2 = 588, 246, 780, 430
    Rain:   Faster    
            DETR      x1, y1, x11, y11 = 990, 455, 1013, 478
            YOLOv6    
            MTSD      x1, y1, x11, y11 = 992, 452, 1018, 476
            x1, y1, x2, y2 = 963, 402, 1100, 535
    Snow:   Faster    x1, y1, x11, y11 = 369, 82, 398, 140    x1, y1, x11, y11 = 558, 126, 585, 154    x1, y1, x11, y11 = 558, 126, 585, 154
            DETR      x1, y1, x11, y11 = 377, 88, 401, 139    x1, y1, x11, y11 = 530, 121, 561, 149
            YOLOv6    
            MTSD      
            x1, y1, x2, y2 = 355, 60, 653, 179
    Fog:    Faster    
            DETR      
            YOLOv6    x1, y1, x11, y11 = 738, 68, 772, 98
            MTSD      x1, y1, x11, y11 = 735, 69, 773, 100
            x1, y1, x2, y2 = 694, 36, 866, 139
    Night:  Faster    
            DETR      x1, y1, x11, y11 = 650, 62, 701, 106
            YOLOv6    
            MTSD      x1, y1, x11, y11 = 651, 63, 698, 108
            x1, y1, x2, y2 = 626, 40, 788, 159
'''
dir = "DETR"
imgname="003602_night.jpg"
imgname2="003602_night_small.jpg"
imgname3="003602_night_big.jpg"
image = cv2.imread("./All/"+imgname)
H, W = image.shape[0], image.shape[1]

text = str("SpeedLimit30")
x1, y1, x11, y11 = 650, 62, 701, 106
c1, c11 = (int(x1), int(y1)), (int(x11), int(y11))
cv2.rectangle(image, c1, c11, (255, 200, 0), thickness=1, lineType=cv2.LINE_AA)
font_thickness = max(2 - 1, 1)
t_size = cv2.getTextSize(text, 0, fontScale=0.45, thickness=font_thickness)[0]
c11 = c1[0] + t_size[0] + 7, c1[1] - t_size[1] - 3
cv2.rectangle(image, c1, c11, (255, 200, 0), -1, cv2.LINE_AA)
cv2.putText(image, text, (c1[0], c1[1] - 2), 0, 0.45,
                    [0, 0, 0], thickness=font_thickness, lineType=cv2.LINE_AA)

# text = str("SpeepLimit40")
# x1, y1, x11, y11 = 523, 120, 558, 149
# c1, c11 = (int(x1), int(y1)), (int(x11), int(y11))
# cv2.rectangle(image, c1, c11, (234, 150, 255), thickness=1, lineType=cv2.LINE_AA)
# font_thickness = max(2 - 1, 1)
# t_size = cv2.getTextSize(text, 0, fontScale=0.45, thickness=font_thickness)[0]
# # c11 = c1[0] + t_size[0] + 7, c1[1] - t_size[1] - 3
# c3 = (int(x11), int(y1))
# c11 = c3[0] - t_size[0] - 7, c3[1] - t_size[1] - 3
# cv2.rectangle(image, c3, c11, (234, 150, 255), -1, cv2.LINE_AA)
# cv2.putText(image, text, (c11[0], c1[1] - 2), 0, 0.45,
#                     [0, 0, 0], thickness=font_thickness, lineType=cv2.LINE_AA)

# text = str("No Parking")
# x1, y1, x11, y11 = 558, 126, 585, 154
# c1, c11 = (int(x1), int(y1)), (int(x11), int(y11))
# cv2.rectangle(image, c1, c11, (52, 56, 255), thickness=1, lineType=cv2.LINE_AA)
# font_thickness = max(2 - 1, 1)
# t_size = cv2.getTextSize(text, 0, fontScale=0.45, thickness=font_thickness)[0]
# c11 = c1[0] + t_size[0] + 7, c1[1] - t_size[1] - 3
# cv2.rectangle(image, c1, c11, (52, 56, 255), -1, cv2.LINE_AA)
# cv2.putText(image, text, (c1[0], c1[1] - 2), 0, 0.45,
#                     [0, 0, 0], thickness=font_thickness, lineType=cv2.LINE_AA)

cv2.imshow("demo", image)
cv2.waitKey(0)
cv2.imwrite("./ALL/"+dir+"/"+imgname3, image)

# 读取原始图片
# img = cv2.imread("./All/20200.jpg")
# 定义矩形位置
x1, y1, x2, y2 = 626, 40, 788, 159
# 截取矩形位置作为小图片
small_img = image[y1:y2, x1:x2]
cv2.imshow("demo", small_img)
cv2.waitKey(0)
# 保存新图片
cv2.imwrite("./ALL/"+dir+"/"+imgname2, small_img)