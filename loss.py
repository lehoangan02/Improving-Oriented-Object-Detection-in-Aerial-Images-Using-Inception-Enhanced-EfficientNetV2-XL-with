import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 1, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 1])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 1])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            loss = F.binary_cross_entropy(pred.masked_select(mask),
                                          target.masked_select(mask),
                                          reduction='mean')
            return loss
        else:
            return 0.

class OffSmoothL1Loss(nn.Module):
    def __init__(self):
        super(OffSmoothL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            loss = F.smooth_l1_loss(pred.masked_select(mask),
                                    target.masked_select(mask),
                                    reduction='mean')
            return loss
        else:
            return 0.

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        # Clamp pred to avoid log(0) or log(negative)
        pred = torch.clamp(pred, min=1e-6, max=1-1e-6)

        loss = 0
        #   print('pred size is {}'.format(pred.size()))
        #   print('pos_inds size is {}'.format(pos_inds.size()))
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        # Check for NaN values
        if torch.isnan(loss):
            print("NaN detected in FocalLoss")
            print(f"pred: {pred}")
            print(f"gt: {gt}")
            print(f"pos_loss: {pos_loss}")
            print(f"neg_loss: {neg_loss}")
            print(f"num_pos: {num_pos}")

        return loss

def isnan(x):
    return x != x

  
class LossAll(torch.nn.Module):
    def __init__(self):
        super(LossAll, self).__init__()
        self.L_hm = FocalLoss()
        self.L_wh =  OffSmoothL1Loss()
        self.L_off = OffSmoothL1Loss()
        self.L_cls_theta = BCELoss()
        self.L_corners = OffSmoothL1Loss()

    def forward(self, pr_decs, gt_batch):
        hm_loss  = self.L_hm(pr_decs['hm'], gt_batch['hm'])
        wh_loss  = self.L_wh(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss = self.L_off(pr_decs['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])
        # if 'corners' in pr_decs:
        #     corners_loss = self.L_corners(pr_decs['corners'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['corners'])
        # else:
        #     corners_loss = 0
        ## add
        cls_theta_loss = self.L_cls_theta(pr_decs['cls_theta'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['cls_theta'])

        if isnan(hm_loss) or isnan(wh_loss) or isnan(off_loss):
            print('hm loss is {}'.format(hm_loss))
            print('wh loss is {}'.format(wh_loss))
            print('off loss is {}'.format(off_loss))
            # print('corners loss is {}'.format(corners_loss))

        # print(f"hm_loss: {hm_loss}")
        # print(f"wh_loss: {wh_loss}")
        # print(f"off_loss: {off_loss}")
        # print(f"cls_theta_loss: {cls_theta_loss}")
        # if 'corners' in pr_decs:
        #     print(f"corners_loss: {corners_loss.item()}")
        # print('-----------------')

        # loss =  hm_loss + wh_loss + off_loss + cls_theta_loss+corners_loss
        loss =  hm_loss + wh_loss + off_loss + cls_theta_loss
        return loss
class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            mse_loss = nn.MSELoss(reduction='mean')
            loss = mse_loss(pred.masked_select(mask),
                                    target.masked_select(mask))
            return loss
        else:
            return 0.
class LossAll_wh_2(torch.nn.Module):
    def __init__(self):
        # print('LossAll_wh_2')
        super().__init__()
        self.L_hm = FocalLoss()
        self.L_wh =  OffSmoothL1Loss()
        self.L_off = OffSmoothL1Loss()
        self.L_cls_theta = BCELoss()
        self.L_corners = OffSmoothL1Loss()

    def forward(self, pr_decs, gt_batch):
        hm_loss  = self.L_hm(pr_decs['hm'], gt_batch['hm'])
        wh_loss  = self.L_wh(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh']) * 2
        off_loss = self.L_off(pr_decs['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])
        ## add
        cls_theta_loss = self.L_cls_theta(pr_decs['cls_theta'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['cls_theta'])

        if isnan(hm_loss) or isnan(wh_loss) or isnan(off_loss):
            print('hm loss is {}'.format(hm_loss))
            print('wh loss is {}'.format(wh_loss))
            print('off loss is {}'.format(off_loss))
            # print('corners loss is {}'.format(corners_loss))

        # print(f"hm_loss: {hm_loss}")
        # print(f"wh_loss: {wh_loss}")
        # print(f"off_loss: {off_loss}")
        # print(f"cls_theta_loss: {cls_theta_loss}")
        # if 'corners' in pr_decs:
        #     print(f"corners_loss: {corners_loss}")
        # print('-----------------')

        loss =  hm_loss + wh_loss + off_loss + cls_theta_loss
        return loss
class LossAll_aux(torch.nn.Module):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.L_hm = FocalLoss()
        self.L_wh =  OffSmoothL1Loss()
        self.L_off = OffSmoothL1Loss()
        self.L_cls_theta = BCELoss()
        self.L_corners = OffSmoothL1Loss()
        self.ratio = ratio

    def forward(self, decs, gt_batch):
        pr_decs, aux_decs = decs
        aux_hm_loss  = self.L_hm(aux_decs['hm'], gt_batch['hm'])
        aux_wh_loss  = self.L_wh(aux_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        aux_off_loss = self.L_off(aux_decs['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])
        ## add
        aux_cls_theta_loss = self.L_cls_theta(aux_decs['cls_theta'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['cls_theta'])
        
        hm_loss  = self.L_hm(pr_decs['hm'], gt_batch['hm'])
        wh_loss  = self.L_wh(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss = self.L_off(pr_decs['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])
       
        ## add
        cls_theta_loss = self.L_cls_theta(pr_decs['cls_theta'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['cls_theta'])

        if isnan(hm_loss) or isnan(wh_loss) or isnan(off_loss):
            print('hm loss is {}'.format(hm_loss))
            print('wh loss is {}'.format(wh_loss))
            print('off loss is {}'.format(off_loss))

        # print(f"hm_loss: {hm_loss.item()}")
        # print(f"wh_loss: {wh_loss.item()}")
        # print(f"off_loss: {off_loss.item()}")
        # print(f"cls_theta_loss: {cls_theta_loss.item()}")
        # if 'corners' in pr_decs:
        #     print(f"corners_loss: {corners_loss.item()}")
        # print('-----------------')
        aux_loss = aux_hm_loss + aux_wh_loss + aux_off_loss + aux_cls_theta_loss
        # print('aux_loss is {}'.format(aux_loss))
        aux_loss = aux_loss * self.ratio
        # print('ratio is {}'.format(self.ratio))
        main_loss =  hm_loss + wh_loss + off_loss + cls_theta_loss
        # print('main_loss is {}'.format(main_loss))
        loss = aux_loss + main_loss
        # print('loss is {}'.format(loss))
        return loss
class FocalLossSeverePunish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        # Clamp pred to avoid log(0) or log(negative)
        pred = torch.clamp(pred, min=1e-6, max=1-1e-6)

        loss = 0
        #   print('pred size is {}'.format(pred.size()))
        #   print('pos_inds size is {}'.format(pos_inds.size()))
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * 1.5
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        # Check for NaN values
        if torch.isnan(loss):
            print("NaN detected in FocalLoss")
            print(f"pred: {pred}")
            print(f"gt: {gt}")
            print(f"pos_loss: {pos_loss}")
            print(f"neg_loss: {neg_loss}")
            print(f"num_pos: {num_pos}")

        return loss
class LossHeatmapOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.L_hm = FocalLossSeverePunish()

    def forward(self, pr_decs, gt_batch):
        hm_loss  = self.L_hm(pr_decs['hm'], gt_batch['hm'])
        
        if isnan(hm_loss):
            print('hm loss is {}'.format(hm_loss))
            print('this is nan')

        loss =  hm_loss
        return loss
class LossHeatmapOnly_aux(torch.nn.Module):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio
        self.L_hm = FocalLossSeverePunish()

    def forward(self, pr_decs, gt_batch):
        pr_decs, aux_decs = pr_decs
        hm_loss  = self.L_hm(pr_decs['hm'], gt_batch['hm'])
        aux_hm_loss  = self.L_hm(aux_decs['hm'], gt_batch['hm'])
        if isnan(hm_loss):
            print('hm loss is {}'.format(hm_loss))
            print('aux_hm_loss is {}'.format(aux_hm_loss))
            print('this is nan')


        loss =  hm_loss + aux_hm_loss * self.ratio
        return loss