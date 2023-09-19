# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))


    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        # 每一个都是append的 有feature map个 每个都是当前这个feature map中3个anchor筛选出的所有的target(3个grid_cell进行预测)
        # tcls: 表示这个target所属的class index
        # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
        # indices: b: 表示这个target属于的image index
        #          a: 表示这个target使用的anchor index
        #          gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
        #          gi: 表示这个网格的左上角x坐标
        # anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        # 依次遍历三个feature map的预测输出pi
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj  初始化target置信度(先全是负样本 后面再筛选正样本赋值)

            n = b.shape[0]  # number of targets
            if n:
                # 精确得到第b张图片的第a个feature map的grid_cell(gi, gj)对应的预测值
                # 用这个预测值与我们筛选的这个grid_cell的真实框进行预测(计算损失)
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                # Regression loss  只计算所有正样本的回归损失
                # 新的公式:  pxy = [-0.5 + cx, 1.5 + cx]    pwh = [0, 4pw]   这个区域内都是正样本
                # Get more positive samples, accelerate convergence and be more stable
                pxy = ps[:, :2].sigmoid() * 2. - 0.5  # 一个归一化操作 和论文里不同
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # 和论文里不同 这里是作者自己提出的公式
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # 这里的tbox[i]中的xy是这个target对当前grid_cell左上角的偏移量[0,1]  而pbox.T是一个归一化的值
                # 就是要用这种方式训练 传回loss 修改梯度 让pbox越来越接近tbox(偏移量)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # Objectness loss stpe1
                # iou.detach()  不会更新iou梯度  iou并不是反向传播的参数 所以不需要反向传播梯度信息
                score_iou = iou.detach().clamp(0).type(tobj.dtype)  # .clamp(0)必须大于等于0
                if self.sort_obj_iou:  # 可以看下官方的解释 我也不是很清楚为什么这里要对iou排序？？？
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                # 预测信息有置信度 但是真实框信息是没有置信度的 所以需要我们认为的给一个标准置信度
                # self.gr是iou ratio [0, 1]  self.gr越大置信度越接近iou  self.gr越小越接近1(人为加大训练难度)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification  只计算所有正样本的分类损失
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # targets 原本负样本是0  这里使用smooth label 就是cn
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp  # 筛选到的正样本对应位置值是cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            # Objectness loss stpe2 置信度损失是用所有样本(正样本 + 负样本)一起计算损失的
            obji = self.BCEobj(pi[..., 4], tobj)
            # 每个feature map的置信度损失权重不同  要乘以相应的权重系数self.balance[i]
            # 一般来说，检测小物体的难度大一点，所以会增加大特征图的损失系数，让模型更加侧重小物体的检测
            lobj += obji * self.balance[i]  # obj loss

            if self.autobalance:
                # 自动更新各个feature map的置信度损失系数
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # 根据超参中的损失权重参数 对各个损失进行平衡  防止总损失被某个损失所左右
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']

        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []

        # gain是为了后面将targets=[na,nt,7]中的归一化了的xywh映射到相对feature map尺度上
        # 7: image_index+class+xywh+anchor_index
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain

        # 需要在3个anchor上都进行训练 所以将标签赋值na=3个  ai代表3个anchor上在所有的target对应的anchor索引 就是用来标记下当前这个target属于哪个anchor
        # [1, 3] -> [3, 1] -> [3, 63]=[na, nt]   三行  第一行63个0  第二行63个1  第三行63个2
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        # [63, 6] [3, 63] -> [3, 63, 6] [3, 63, 1] -> [3, 63, 7]  7: [image_index+class+xywh+anchor_index]
        # 对每一个feature map: 这一步是将target复制三份 对应一个feature map的三个anchor
        # 先假设所有的target对三个anchor都是正样本(复制三份) 再进行筛选  并将ai加进去标记当前是哪个anchor的target
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        # 这两个变量是用来扩展正样本的 因为预测框预测到target有可能不止当前的格子预测到了
        # 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
        g = 0.5  # bias  中心偏移  用来衡量target中心点离哪个格子更近
        # 以自身 + 周围左上右下4个网格 = 5个网格  用来计算offsets
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        # 遍历三个feature 筛选每个feature map(包含batch张图片)的每个anchor的正样本
        for i in range(self.nl):
            anchors = self.anchors[i]

            # gain: 保存每个输出feature map的宽高 -> gain[2:6]=gain[whwh]
            # [1, 1, 1, 1, 1, 1, 1] -> [1, 1, 112, 112, 112,112, 1]=image_index+class+xywh+anchor_index
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # t = [3, 63, 7]  将target中的xywh的归一化尺度放缩到相对当前feature map的坐标尺度
            #     [3, 63, image_index+class+xywh+anchor_index]
            t = targets * gain

            if nt:
                # Matches
                # t=[na, nt, 7]   t[:, :, 4:6]=[na, nt, 2]=[3, 63, 2]
                # anchors[:, None]=[na, 1, 2]
                # r=[na, nt, 2]=[3, 63, 2]
                # 当前feature map的3个anchor的所有正样本(没删除前是所有的targets)与三个anchor的宽高比(w/w  h/h)
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio

                # 筛选条件  GT与anchor的宽比或高比超过一定的阈值 就当作负样本
                # torch.max(r, 1. / r)=[3, 63, 2] 筛选出宽比w1/w2 w2/w1 高比h1/h2 h2/h1中最大的那个
                # .max(2)返回宽比 高比两者中较大的一个值和它的索引  [0]返回较大的一个值
                # j: [3, 63]  False: 当前gt是当前anchor的负样本  True: 当前gt是当前anchor的正样本
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))

                # yolov3 v4的筛选方法: wh_iou  GT与anchor的wh_iou超过一定的阈值就是正样本
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # 根据筛选条件j, 过滤负样本, 得到当前feature map上三个anchor的所有正样本t(batch_size张图片)
                # t: [3, 63, 7] -> [126, 7]  [num_Positive_sample, image_index+class+xywh+anchor_index]
                t = t[j]  # filter

                # Offsets
                # Offsets 筛选当前格子周围格子 找到2个离target中心最近的两个格子  可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
                # 除了target所在的当前格子外, 还有2个格子对目标进行检测(计算损失) 也就是说一个目标需要3个格子去预测(计算损失)
                # 首先当前格子是其中1个 再从当前格子的上下左右四个格子中选择2个 用这三个格子去预测这个目标(计算损失)
                # feature map上的原点在左上角 向右为x轴正坐标 向下为y轴正坐标
                gxy = t[:, 2:4]  # grid xy 取target中心的坐标xy(相对feature map左上角的坐标)
                gxi = gain[[2, 3]] - gxy  # inverse  得到target中心点相对于右下角的坐标  gain[[2, 3]]为当前feature map的wh
                # 筛选中心坐标 距离当前grid_cell的左、上方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # j: [126] bool 如果是True表示当前target中心点所在的格子的左边格子也对该target进行回归(后续进行计算损失)
                # k: [126] bool 如果是True表示当前target中心点所在的格子的上边格子也对该target进行回归(后续进行计算损失)
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # 筛选中心坐标 距离当前grid_cell的右、下方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # l: [126] bool 如果是True表示当前target中心点所在的格子的右边格子也对该target进行回归(后续进行计算损失)
                # m: [126] bool 如果是True表示当前target中心点所在的格子的下边格子也对该target进行回归(后续进行计算损失)
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                # j: [5, 126]  torch.ones_like(j): 当前格子, 不需要筛选全是True  j, k, l, m: 左上右下格子的筛选结果
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # 得到筛选后所有格子的正样本 格子数<=3*126 都不在边上等号成立
                # t: [126, 7] -> 复制5份target[5, 126, 7]  分别对应当前格子和左上右下格子5个格子
                # j: [5, 126] + t: [5, 126, 7] => t: [378, 7] 理论上是小于等于3倍的126 当且仅当没有边界的格子等号成立
                t = t.repeat((5, 1, 1))[j]
                # torch.zeros_like(gxy)[None]: [1, 126, 2]   off[:, None]: [5, 1, 2]  => [5, 126, 2]
                # j筛选后: [378, 2]  得到所有筛选后的网格的中心相对于这个要预测的真实框所在网格边界（左右上下边框）的偏移量
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()  # 预测真实框的网格所在的左上角坐标(有左上右下的网格)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            # b: image index  a: anchor index  gj: 网格的左上角y坐标  gi: 网格的左上角x坐标
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            # tbix: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors 对应的所有anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch















# def box_iou_v5(box1, box2, x1y1x2y2=True):
#     # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
#     """
#     Return intersection-over-union (Jaccard index) of boxes.
#     Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
#     Arguments:
#         box1 (Tensor[N, 4])
#         box2 (Tensor[M, 4])
#     Returns:
#         iou (Tensor[N, M]): the NxM matrix containing the pairwise
#             IoU values for every element in boxes1 and boxes2
#     """
#
#     def box_area(box, x1y1x2y2=True):
#         # box = 4xn
#         if x1y1x2y2:
#             return (box[2] - box[0]) * (box[3] - box[1])
#         else:
#             b_x1, b_x2 = box[0] - box[2] / 2, box[0] + box[2] / 2
#             b_y1, b_y2 = box[1] - box[3] / 2, box[1] + box[3] / 2
#             return (b_x2 - b_x1) * (b_y2 - b_y1)
#     area1 = box_area(box1.T, x1y1x2y2)
#     area2 = box_area(box2.T, x1y1x2y2)
#
#     # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
#     inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
#     return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
#
#
# def IoG(gt_box, pre_box):
#     inter_xmin = torch.max(gt_box[:, 0], pre_box[:, 0])
#     inter_ymin = torch.max(gt_box[:, 1], pre_box[:, 1])
#     inter_xmax = torch.min(gt_box[:, 2], pre_box[:, 2])
#     inter_ymax = torch.min(gt_box[:, 3], pre_box[:, 3])
#     Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
#     Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
#     I = Iw * Ih
#     G = ((gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])).clamp(1e-6)
#     return I / G
#
# import numpy as np
# def smooth_ln(x, deta=0.5):
#     return torch.where(
#         torch.le(x, deta),
#         -torch.log(1 - x),
#         ((x - deta) / (1 - deta)) - np.log(1 - deta)
#     )
#
# # YU 添加了detach，减小了梯度对gpu的占用
# def repulsion_loss_torch(pbox, gtbox, deta=0.5, pnms=0.1, gtnms=0.1, x1x2y1y2=False):
#     repgt_loss = 0.0
#     repbox_loss = 0.0
#     pbox = pbox.detach()
#     gtbox = gtbox.detach()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     gtbox_cpu = gtbox.cuda().data.cpu().numpy()
#     pgiou = box_iou_v5(pbox, gtbox, x1y1x2y2=x1x2y1y2)
#     pgiou = pgiou.cuda().data.cpu().numpy()
#     ppiou = box_iou_v5(pbox, pbox, x1y1x2y2=x1x2y1y2)
#     ppiou = ppiou.cuda().data.cpu().numpy()
#     # t1 = time.time()
#     len = pgiou.shape[0]
#     for j in range(len):
#         for z in range(j, len):
#             ppiou[j, z] = 0
#             # if int(torch.sum(gtbox[j] == gtbox[z])) == 4:
#             # if int(torch.sum(gtbox_cpu[j] == gtbox_cpu[z])) == 4:
#             # if int(np.sum(gtbox_numpy[j] == gtbox_numpy[z])) == 4:
#             if (gtbox_cpu[j][0]==gtbox_cpu[z][0]) and (gtbox_cpu[j][1]==gtbox_cpu[z][1]) and (gtbox_cpu[j][2]==gtbox_cpu[z][2]) and (gtbox_cpu[j][3]==gtbox_cpu[z][3]):
#                 pgiou[j, z] = 0
#                 pgiou[z, j] = 0
#                 ppiou[z, j] = 0
#
#     # t2 = time.time()
#     # print("for cycle cost time is: ", t2 - t1, "s")
#     pgiou = torch.from_numpy(pgiou).cuda().detach()
#     ppiou = torch.from_numpy(ppiou).cuda().detach()
#     # repgt
#     max_iou, argmax_iou = torch.max(pgiou, 1)
#     pg_mask = torch.gt(max_iou, gtnms)
#     num_repgt = pg_mask.sum()
#     if num_repgt > 0:
#         iou_pos = pgiou[pg_mask, :]
#         max_iou_sec, argmax_iou_sec = torch.max(iou_pos, 1)
#         pbox_sec = pbox[pg_mask, :]
#         gtbox_sec = gtbox[argmax_iou_sec, :]
#         IOG = IoG(gtbox_sec, pbox_sec)
#         repgt_loss = smooth_ln(IOG, deta)
#         repgt_loss = repgt_loss.mean()
#
#     # repbox
#     pp_mask = torch.gt(ppiou, pnms)  # 防止nms为0, 因为如果为0,那么上面的for循环就没有意义了 [N x N] error
#     num_pbox = pp_mask.sum()
#     if num_pbox > 0:
#         repbox_loss = smooth_ln(ppiou, deta)
#         repbox_loss = repbox_loss.mean()
#     # mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
#     # print(mem)
#     torch.cuda.empty_cache()
#
#     return repgt_loss, repbox_loss
#
#
# import math
# class reSwanLoss(nn.Module):
#     def __init__(self, loss_fcn):
#         super(reSwanLoss, self).__init__()
#         self.loss_fcn = loss_fcn
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = 'none'  # required to apply SL to each element
#
#     def forward(self, pred, true, auto_iou=0.5):
#         loss = self.loss_fcn(pred, true)
#         if auto_iou < 0.2:
#             auto_iou = 0.2
#         b1 = true <= auto_iou - 0.1
#         a1 = 1.0
#         b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
#         a2 = math.exp(auto_iou)
#         b3 = true >= auto_iou
#         a3 = torch.exp(-(true - 1.0))
#         modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
#         loss *= modulating_weight
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:  # 'none'
#             return loss
#
#
# def Wasserstein(box1, box2, x1y1x2y2=True):
#     box2 = box2.T
#     if x1y1x2y2:
#         b1_cx, b1_cy = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
#         b1_w, b1_h = box1[2] - box1[0], box1[3] - box1[1]
#         b2_cx, b2_cy = (box2[0] + box2[0]) / 2, (box2[1] + box2[3]) / 2
#         b1_w, b1_h = box2[2] - box2[0], box2[3] - box2[1]
#     else:
#         b1_cx, b1_cy, b1_w, b1_h = box1[0], box1[1], box1[2], box1[3]
#         b2_cx, b2_cy, b2_w, b2_h = box2[0], box2[1], box2[2], box2[3]
#     cx_L2Norm = torch.pow((b1_cx - b2_cx), 2)
#     cy_L2Norm = torch.pow((b1_cy - b2_cy), 2)
#     p1 = cx_L2Norm + cy_L2Norm
#     w_FroNorm = torch.pow((b1_w - b2_w)/2, 2)
#     h_FroNorm = torch.pow((b1_h - b2_h)/2, 2)
#     p2 = w_FroNorm + h_FroNorm
#     return p1 + p2
#
#
#
# class ComputeNWDLoss:
#     # Compute losses
#     def __init__(self, model, autobalance=False):
#         self.sort_obj_iou = False
#         device = next(model.parameters()).device  # get model device
#         h = model.hyp  # hyperparameters
#
#         # Define criteria
#         BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
#         BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
#
#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
#
#         # update c: 3.0, u: 1.0
#         self.C = 3.0
#         # reswan loss
#         self.u = 1.0
#         if self.u > 0:
#             BCEcls, BCEobj = reSwanLoss(BCEcls), reSwanLoss(BCEobj)
#         # Focal loss
#         g = h['fl_gamma']  # focal loss gamma
#         if g > 0:
#             BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
#
#         det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
#         self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
#         self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
#         for k in 'na', 'nc', 'nl', 'anchors':
#             setattr(self, k, getattr(det, k))
#
#     def __call__(self, p, targets):  # predictions, targets, model
#         device = targets.device
#         lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
#         lrepBox, lrepGT = torch.zeros(1, device=device), torch.zeros(1, device=device)  # update
#         tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
#         auto_iou = 0.5
#         # Losses
#         for i, pi in enumerate(p):  # layer index, layer predictions
#             b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#             tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
#
#             n = b.shape[0]  # number of targets
#             if n:
#                 ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
#
#                 # Regression
#                 pxy = ps[:, :2].sigmoid() * 2 - 0.5
#                 pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1)  # predicted box
#                 iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
#                 # nwd update
#                 nwd = torch.exp(-torch.pow(Wasserstein(pbox.T, tbox[i], x1y1x2y2=False), 1 / 2) / self.C)
#                 auto_iou = iou.mean()
#                 lbox += 0.8 * (1.0 - iou).mean() + 0.2 * (1.0 - nwd).mean()
#                 # lbox += (1.0 - iou).mean()  # iou loss origin
#
#                 # Objectness
#                 score_iou = iou.detach().clamp(0).type(tobj.dtype)
#                 if self.sort_obj_iou:
#                     sort_id = torch.argsort(score_iou)
#                     b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
#                 tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio
#
#                 # Classification
#                 if self.nc > 1:  # cls loss (only if multiple classes)
#                     t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
#                     t[range(n), tcls[i]] = self.cp
#                     if self.u > 0:
#                         lcls += self.BCEcls(ps[:, 5:], t, auto_iou)  # BCE
#                     else:
#                         lcls += self.BCEcls(ps[:, 5:], t)  # BCE
#                     # lcls += self.BCEcls(ps[:, 5:], t)  # BCE原始
#
#                 # Replusion Loss update
#                 dic = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [],
#                        13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [],
#                        25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}
#                 for indexs, value in enumerate(b):
#                     # print(indexs, value)
#                     dic[int(value)].append(indexs)
#                 # print('dic', dic)
#                 bts = 0  # update
#                 deta = 0.5  # smooth_ln parameter
#                 Rp_nms = 0.1  # RepGT loss nms
#                 _lrepGT = 0.0
#                 _lrepBox = 0.0
#                 for id, indexs in dic.items():  # id = batch_name  indexs = target_id
#                     if indexs:
#                         lrepgt, lrepbox = repulsion_loss_torch(pbox[indexs], tbox[i][indexs], deta=deta, pnms=Rp_nms,
#                                                                gtnms=Rp_nms)
#                         _lrepGT += lrepgt
#                         _lrepBox += lrepbox
#                         bts += 1
#                 if bts > 0:
#                     _lrepGT /= bts
#                     _lrepBox /= bts
#                 lrepGT += _lrepGT
#                 lrepBox += _lrepBox
#
#                 # Append targets to text file
#                 # with open('targets.txt', 'a') as file:
#                 #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
#
#             # update
#             if self.u > 0:
#                 obji = self.BCEobj(pi[..., 4], tobj, auto_iou)
#             else:
#                 obji = self.BCEobj(pi[..., 4], tobj)
#             # obji = self.BCEobj(pi[..., 4], tobj)
#             lobj += obji * self.balance[i]  # obj loss
#             if self.autobalance:
#                 self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
#
#         if self.autobalance:
#             self.balance = [x / self.balance[self.ssi] for x in self.balance]
#         lbox *= self.hyp['box']
#         lobj *= self.hyp['obj']
#         lcls *= self.hyp['cls']
#         # lrep = self.hyp['alpha'] * lrepGT / 3.0 + self.hyp['beta'] * lrepBox / 3.0
#         # alpha-0.01: RepGT loss gain 0.04, init: 0.233 * 2
#         # beta-0.1: RepBox loss gain 0.6, init: 0.0222 * 2
#         lrep = 0.01 * lrepGT / 3.0 + 0.1 * lrepBox / 3.0
#         bs = tobj.shape[0]  # batch size, gpu
#
#         # no lrep, loss
#         loss = lbox + lobj + lcls + lrep
#         return loss * bs, torch.cat((lbox, lobj, lcls)).detach()
#
#     def build_targets(self, p, targets):
#         # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         tcls, tbox, indices, anch = [], [], [], []
#         gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
#         ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
#
#         g = 0.5  # bias
#         off = torch.tensor([[0, 0],
#                             [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
#                             # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#                             ], device=targets.device).float() * g  # offsets
#
#         for i in range(self.nl):
#             anchors = self.anchors[i]
#             gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
#
#             # Match targets to anchors
#             t = targets * gain
#             if nt:
#                 # Matches
#                 r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
#                 j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
#                 # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#                 t = t[j]  # filter
#
#                 # Offsets
#                 gxy = t[:, 2:4]  # grid xy
#                 gxi = gain[[2, 3]] - gxy  # inverse
#                 j, k = ((gxy % 1 < g) & (gxy > 1)).T
#                 l, m = ((gxi % 1 < g) & (gxi > 1)).T
#                 j = torch.stack((torch.ones_like(j), j, k, l, m))
#                 t = t.repeat((5, 1, 1))[j]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#             else:
#                 t = targets[0]
#                 offsets = 0
#
#             # Define
#             b, c = t[:, :2].long().T  # image, class
#             gxy = t[:, 2:4]  # grid xy
#             gwh = t[:, 4:6]  # grid wh
#             gij = (gxy - offsets).long()
#             gi, gj = gij.T  # grid xy indices
#
#             # Append
#             a = t[:, 6].long()  # anchor indices
#             indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
#             tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#             anch.append(anchors[a])  # anchors
#             tcls.append(c)  # class
#
#         return tcls, tbox, indices, anch




def Wasserstein(box1, box2, x1y1x2y2=True):
    box2 = box2.T
    if x1y1x2y2:
        b1_cx, b1_cy = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
        b1_w, b1_h = box1[2] - box1[0], box1[3] - box1[1]
        b2_cx, b2_cy = (box2[0] + box2[0]) / 2, (box2[1] + box2[3]) / 2
        b2_w, b2_h = box2[2] - box2[0], box2[3] - box2[1]
    else:
        b1_cx, b1_cy, b1_w, b1_h = box1[0], box1[1], box1[2], box1[3]
        b2_cx, b2_cy, b2_w, b2_h = box2[0], box2[1], box2[2], box2[3]
    cx_L2Norm = torch.pow((b1_cx - b2_cx), 2)
    cy_L2Norm = torch.pow((b1_cy - b2_cy), 2)
    p1 = cx_L2Norm + cy_L2Norm
    w_FroNorm = torch.pow((b1_w - b2_w)/2, 2)
    h_FroNorm = torch.pow((b1_h - b2_h)/2, 2)
    p2 = w_FroNorm + h_FroNorm
    return p1 + p2


class ComputeNWDLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # update c: 3.0, u: 1.0
        self.C = 3.0
        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lrepBox, lrepGT = torch.zeros(1, device=device), torch.zeros(1, device=device)  # update
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        auto_iou = 0.5
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # nwd update
                nwd = torch.exp(-torch.pow(Wasserstein(pbox.T, tbox[i], x1y1x2y2=False), 1 / 2) / self.C)
                # auto_iou = iou.mean()
                lbox += 0.8 * (1.0 - iou).mean() + 0.2 * (1.0 - nwd).mean()
                # lbox += (1.0 - iou).mean()  # iou loss origin

                # Objectness
                # Objectness loss stpe1
                # iou.detach()  不会更新iou梯度  iou并不是反向传播的参数 所以不需要反向传播梯度信息
                score_iou = iou.detach().clamp(0).type(tobj.dtype)  # .clamp(0)必须大于等于0
                if self.sort_obj_iou:  # 可以看下官方的解释 我也不是很清楚为什么这里要对iou排序？？？
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                # 预测信息有置信度 但是真实框信息是没有置信度的 所以需要我们认为的给一个标准置信度
                # self.gr是iou ratio [0, 1]  self.gr越大置信度越接近iou  self.gr越小越接近1(人为加大训练难度)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification  只计算所有正样本的分类损失
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # targets 原本负样本是0  这里使用smooth label 就是cn
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp  # 筛选到的正样本对应位置值是cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

                # Objectness loss stpe2 置信度损失是用所有样本(正样本 + 负样本)一起计算损失的
            obji = self.BCEobj(pi[..., 4], tobj)
            # 每个feature map的置信度损失权重不同  要乘以相应的权重系数self.balance[i]
            # 一般来说，检测小物体的难度大一点，所以会增加大特征图的损失系数，让模型更加侧重小物体的检测
            lobj += obji * self.balance[i]  # obj loss

            if self.autobalance:
                # 自动更新各个feature map的置信度损失系数
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # 根据超参中的损失权重参数 对各个损失进行平衡  防止总损失被某个损失所左右
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']

        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch