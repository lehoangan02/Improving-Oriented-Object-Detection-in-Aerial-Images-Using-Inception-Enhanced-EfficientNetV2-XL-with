import torch.nn.functional as F
import torch

class DecDecoder(object):
    def __init__(self, K, conf_thresh, num_classes):
        self.K = K
        self.conf_thresh = conf_thresh
        self.num_classes = num_classes

    def _topk(self, scores):
        batch, cat, height, width = scores.size()
        # calculate top k for each class
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)
        #topk_scores shape: [batch, cat, K]
        #topk_inds shape: [batch, cat, K]
        #topk_scores shape: [batch, cat, K]
        #topk_inds shape: [batch, cat, K]

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()
        # calculate top k for all classes of the chosen top k for each class
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        #topk_score shape: [batch, K]
        #topk_ind shape: [batch, K]

        # print(f"topk_score size: {topk_score.size()}")
        # print(f"topk_ind size: {topk_ind.size()}")

        topk_clses = (topk_ind // self.K).int()
        #topk_clses shape: [batch, K]


        #topk_inds.view(batch, -1, 1) shape: [batch, cat*K, 1]
        #topk_ys.view(batch, -1, 1) shape: [batch, cat*K, 1]
        #topk_xs.view(batch, -1, 1) shape: [batch, cat*K, 1]
        topk_inds = self._gather_feat( topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        #feat shape: [batch, cat*K]
        dim = feat.size(2)
        #dim = 1
        #ind shape: [batch, K]
        #ind.unsqueeze(2) shape: [batch, K, 1]
        #ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim) shape: [batch, K,  1]
        #dim = cat*K
        #ind shape: [batch, K]
        
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = torch.gather(feat, 1, ind) #shape: [batch, K, 1], gather the top K scores for each class
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

    def ctdet_decode(self, pr_decs):
        heat = pr_decs['hm']
        wh = pr_decs['wh']
        reg = pr_decs['reg']
        cls_theta = pr_decs['cls_theta']

        batch, c, height, width = heat.size()
        heat = self._nms(heat)

        scores, inds, clses, ys, xs = self._topk(heat)
        reg = self._tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, self.K, 2)
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        clses = clses.view(batch, self.K, 1).float()
        scores = scores.view(batch, self.K, 1)
        wh = self._tranpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, self.K, 10)
        # add
        cls_theta = self._tranpose_and_gather_feat(cls_theta, inds)
        cls_theta = cls_theta.view(batch, self.K, 1)
        mask = (cls_theta>0.8).float().view(batch, self.K, 1)
        #
        # wh shape: [batch, K, 10]
        # 0:1: tt_x
        # 1:2: tt_y
        # 2:3: rr_x
        # 3:4: rr_y
        # 4:5: bb_x
        # 5:6: bb_y
        # 6:7: ll_x
        # 7:8: ll_y
        # 8:9: w
        # 9:10: h
        tt_x = (xs+wh[..., 0:1])*mask + (xs)*(1.-mask)
        tt_y = (ys+wh[..., 1:2])*mask + (ys-wh[..., 9:10]/2)*(1.-mask)
        rr_x = (xs+wh[..., 2:3])*mask + (xs+wh[..., 8:9]/2)*(1.-mask)
        rr_y = (ys+wh[..., 3:4])*mask + (ys)*(1.-mask)
        bb_x = (xs+wh[..., 4:5])*mask + (xs)*(1.-mask)
        bb_y = (ys+wh[..., 5:6])*mask + (ys+wh[..., 9:10]/2)*(1.-mask)
        ll_x = (xs+wh[..., 6:7])*mask + (xs-wh[..., 8:9]/2)*(1.-mask)
        ll_y = (ys+wh[..., 7:8])*mask + (ys)*(1.-mask)
        #
        detections = torch.cat([xs,                      # cen_x
                                ys,                      # cen_y
                                tt_x,
                                tt_y,
                                rr_x,
                                rr_y,
                                bb_x,
                                bb_y,
                                ll_x,
                                ll_y,
                                scores,
                                clses],
                               dim=2)

        index = (scores>self.conf_thresh).squeeze(0).squeeze(1)
        detections = detections[:,index,:]
        return detections.data.cpu().numpy()