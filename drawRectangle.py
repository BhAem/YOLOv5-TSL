import os

import cv2
import torch
import numpy as np

# 760, 480, 950, 580    5 5
#
# 360, 60, 675, 160     2.2  2.2
#
# 600, 250, 750, 400
#
# 380, 150, 490, 210
# 770, 150, 859, 210
#
# 600, 220, 820, 300    3.5

# 读取原始图片
img = cv2.imread("./2/YOLOv8/w-77.jpg")
# 定义矩形位置
x1, y1, x2, y2 = 600, 220, 820, 300
# 截取矩形位置作为小图片
small_img = img[y1:y2, x1:x2]
# 画上黄色边框
cv2.rectangle(small_img, (0,0), (x2-x1,y2-y1), (0,255,255), thickness=8)
# 放大小图片并放置在原始图片的左下角
resized_small_img = cv2.resize(small_img, None, fx=3.5, fy=3.5, interpolation=cv2.INTER_LINEAR)
h, w, _ = resized_small_img.shape
# img[img.shape[0]-h:, img.shape[1]-w:] = resized_small_img
img[img.shape[0]-h:, :w] = resized_small_img

# x1, y1, x2, y2 = 770, 150, 859, 210
# # 截取矩形位置作为小图片
# small_img2 = img[y1:y2, x1:x2]
# # 画上黄色边框
# cv2.rectangle(small_img2, (0,0), (x2-x1,y2-y1), (0,255,255), thickness=8)
# # 放大小图片并放置在原始图片的左下角
# resized_small_img2 = cv2.resize(small_img2, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
# h, w, _ = resized_small_img2.shape
# img[img.shape[0]-h:, img.shape[1]-w:] = resized_small_img2

# 保存新图片
cv2.imwrite("./2/YOLOv8/w-77_h.jpg", img)




# night_img = r"D:\Users\11939\Desktop\yolov5-6.0\000\000\TrafficSignLight9-daytime\images\dfsn2\1779.jpg"
# snow_img = r"D:\Users\11939\Desktop\yolov5-6.0\000\000\TrafficSignLight9-daytime\images\dfsn2\19123.jpg"
# normal_img = r"D:\Users\11939\Desktop\yolov5-6.0\000\000\TrafficSignLight9-daytime\images\dfsn2\20200.jpg"
# fog_img = r"D:\Users\11939\Desktop\yolov5-6.0\000\000\TrafficSignLight9-daytime\images\dfsn2\BD_Baidu_117.png"
# rain_img = r"D:\Users\11939\Desktop\yolov5-6.0\000\000\TrafficSignLight9-daytime\images\dfsn2\w-77.jpg"
#
# image = rain_img
#
# image = cv2.imread(image)
# H, W = image.shape[0], image.shape[1]
#
# x0, y0, w0, h0 = 0.667368, 0.360449, 0.0357895, 0.0196353
# x0, y0, w0, h0 = x0*W, y0*H, w0*W, h0*H
# x0, y0, x00, y00 = x0 - w0/2, y0 - h0/2, x0 + w0/2, y0 + h0/2
# c0, c00 = (int(x0), int(y0)), (int(x00), int(y00))
# cv2.rectangle(image, c0, c00, (201, 150, 255), thickness=2, lineType=cv2.LINE_AA)
# font_thickness = max(2 - 1, 1)
# t_size = cv2.getTextSize(str("GreenLight"), 0, fontScale=0.45, thickness=font_thickness)[0]
# c00 = c0[0] + t_size[0] + 7, c0[1] - t_size[1] - 3
# cv2.rectangle(image, c0, c00, (201, 150, 255), -1, cv2.LINE_AA)
# cv2.putText(image, str("GreenLight"), (c0[0], c0[1] - 2), 0, 0.45,
#                     [255, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

# x1, y1, w1, h1 = 0.91851, 0.497326, 0.0279395, 0.0695187
# x1, y1, w1, h1 = x1*W, y1*H, w1*W, h1*H
# x1, y1, x11, y11 = x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2
# c1, c11 = (int(x1), int(y1)), (int(x11), int(y11))
# cv2.rectangle(image, c1, c11, (54, 57, 254), thickness=2, lineType=cv2.LINE_AA)
# font_thickness = max(2 - 1, 1)
# t_size = cv2.getTextSize(str("No Parking"), 0, fontScale=0.45, thickness=font_thickness)[0]
# c11 = c1[0] + t_size[0] + 7, c1[1] - t_size[1] - 3
# cv2.rectangle(image, c1, c11, (54, 57, 254), -1, cv2.LINE_AA)
# cv2.putText(image, str("No Parking"), (c1[0], c1[1] - 2), 0, 0.45,
#                     [255, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

x2, y2, w2, h2 = 0.636203, 0.279167, 0.0341981, 0.0625
x2, y2, w2, h2 = x2*W, y2*H, w2*W, h2*H
x2, y2, x22, y22 = x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2
c2, c22 = (int(x2), int(y2)), (int(x22), int(y22))
cv2.rectangle(image, c2, c22, (238, 24, 0), thickness=2, lineType=cv2.LINE_AA)
font_thickness = max(2 - 1, 1)
t_size = cv2.getTextSize(str("Speed Limit 40"), 0, fontScale=0.45, thickness=font_thickness)[0]
c3 = (int(x22), int(y2))
c22 = c3[0] - t_size[0] - 7, c3[1] - t_size[1] - 3
cv2.rectangle(image, c3, c22, (238, 24, 0), -1, cv2.LINE_AA)
cv2.putText(image, str("Speed Limit 40"), (c22[0], c2[1] - 2), 0, 0.45,
                    [255, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

cv2.imwrite(os.path.join(r'D:\Users\11939\Desktop\2\YOLOv6', 'w-77.jpg'), image)









# def draw_bboxes(image, bboxes, line_thickness):
#     name = ["KeepRight: 0.98", "NoGoodVehicles: 0.82", "GreenLight: 0.90", "GreenLight: 0.88"]
#     colors = [(0, 151, 255), (0, 255, 255), (255, 0, 255), (255, 0, 255)]
#     line_thickness = line_thickness or round(0.001 * (image.shape[0] + image.shape[1]) * 0.2) + 1
#     for (x1, y1, x2, y2, cls_id) in bboxes:
#         cls_id = int(cls_id)
#         c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
#         cv2.rectangle(image, c1, c2, colors[cls_id], thickness=line_thickness, lineType=cv2.LINE_AA)
#         font_thickness = max(line_thickness - 1, 1)
#         t_size = cv2.getTextSize(str(name[cls_id]), 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
#         c2 = c1[0] + t_size[0] + 7, c1[1] - t_size[1] - 3
#         cv2.rectangle(image, c1, c2, colors[cls_id], -1, cv2.LINE_AA)
#         cv2.putText(image, name[cls_id], (c1[0], c1[1] - 2), 0, line_thickness / 3,
#                     [0, 0, 0], thickness=font_thickness, lineType=cv2.LINE_AA)
#     return image
#
# def xywh2xyxy(x):
#     # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
#     y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
#     y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
#     y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
#     return y
#
# coor = np.array([[0.145412, 0.311496, 0.037030, 0.062620, 0],
#                  [0.773799, 0.262524, 0.033869, 0.062620, 1],
#                  [0.410975, 0.049484, 0.029472, 0.074019, 2],
#                  [0.564418, 0.048237, 0.027601, 0.069860, 3]])
#
# image = cv2.imread("./drawRect/00020.jpg")
# H, W = image.shape[:2]
# coor[:, 0] *= W
# coor[:, 1] *= H
# coor[:, 2] *= W
# coor[:, 3] *= H
# coor = xywh2xyxy(coor)
#
# images = os.listdir('./drawRect')
# for img in images:
#     image = cv2.imread(os.path.join('./drawRect', img))
#     image = draw_bboxes(image, coor, 1)
#     cv2.imwrite(os.path.join('./dres', img), image)
#     # cv2.imshow("demo", img)
#     # cv2.waitKey(0)
