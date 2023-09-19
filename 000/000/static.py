import os
import cv2

img_path1 = "./images/train"
img_path2 = "./images/test"

txt_path1 = "./labels/train"
txt_path2 = "./labels/test"

imgs = [s.split('.')[0] for s in os.listdir(img_path1)]
txts = [s.split('.')[0] for s in os.listdir(txt_path1)]

total = 0
A, B, C = 0, 0, 0
cnt = 0

for txt in txts:
    print("train: ", total)
    img = cv2.imread(os.path.join(img_path1, txt+'.jpg'))
    H, W = img.shape[:2]
    f = open(os.path.join(txt_path1, txt+'.txt'))
    file = f.readlines()
    for i in file:
        i = i.strip('\n')
        x, y, w, h = i.split(' ')[1:]
        x, y, w, h = float(x), float(y), float(w), float(h)
        x, y, w, h = x*W, y*H, w*W, h*H
        x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
        area = (y2 - y1) * (x2 - x1)
        if area < 32 * 32:
            A += 1
        elif area < 96 * 96:
            B += 1
        else:
            C += 1
        total += 1


imgs = [s.split('.')[0] for s in os.listdir(img_path2)]
txts = [s.split('.')[0] for s in os.listdir(txt_path2)]

for txt in txts:
    print("test: ", total)
    img = cv2.imread(os.path.join(img_path2, txt+'.jpg'))
    H, W = img.shape[:2]
    f = open(os.path.join(txt_path2, txt+'.txt'))
    file = f.readlines()
    for i in file:
        i = i.strip('\n')
        x, y, w, h = i.split(' ')[1:]
        x, y, w, h = float(x), float(y), float(w), float(h)
        x, y, w, h = x*W, y*H, w*W, h*H
        x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
        area = (y2 - y1) * (x2 - x1)
        if area < 32 * 32:
            A += 5
        elif area < 96 * 96:
            B += 5
        else:
            C += 5
        total += 5

print(A/total, B/total, C/total)