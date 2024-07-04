import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.fpnseg import VGG16, ResNet, Bottleneck
from utils.boxlist import BoxList, boxlist_nms, remove_small_box, cat_boxlist
from model.utils.losses import FCOSLoss

def VGG16(in_channels=3):
    """Constructs a VGG16 model.
    Args:
        in_channel (int) : Set the input channel of the model
    """
    model = VGG16(in_channels)
    return model

def ResNet50(in_channel=3, pretrained=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 5, 3], in_channel=in_channel, pretrained=pretrained)
    return model

def ResNet101(in_channel=3, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        in_channel (int) : Set the input channel of the model
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], in_channel=in_channel, pretrained=pretrained)
    return model


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale


def init_conv_kaiming(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, a=1)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_conv_std(module, std=0.01):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=std)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class FCOSPostprocessor(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.threshold = opt.threshold
        self.top_n = opt.top_n
        self.nms_threshold = opt.nms_threshold
        self.post_top_n = opt.post_top_n
        self.min_size = opt.target_min_size
        self.n_class = opt.n_class

    def forward_single_feature_map(self, location, cls_pred, box_pred, center_pred, image_sizes):
        batch, channel, height, width = cls_pred.shape

        cls_pred = cls_pred.view(batch, channel, height, width).permute(0, 2, 3, 1)
        cls_pred = cls_pred.reshape(batch, -1, channel).sigmoid()

        box_pred = box_pred.view(batch, 4, height, width).permute(0, 2, 3, 1)
        box_pred = box_pred.reshape(batch, -1, 4)

        center_pred = center_pred.view(batch, 1, height, width).permute(0, 2, 3, 1)
        center_pred = center_pred.reshape(batch, -1).sigmoid()

        candid_ids = cls_pred > self.threshold
        top_ns = candid_ids.contiguous().view(batch, -1).sum(1)
        top_ns = top_ns.clamp(max=self.top_n)

        cls_pred = cls_pred * center_pred[:, :, None]

        results = []

        for i in range(batch):
            cls_p = cls_pred[i]
            candid_id = candid_ids[i]
            cls_p = cls_p[candid_id]
            candid_nonzero = candid_id.nonzero()
            box_loc = candid_nonzero[:, 0]
            class_id = candid_nonzero[:, 1] + 1

            box_p = box_pred[i]
            box_p = box_p[box_loc]
            loc = location[box_loc]

            top_n = top_ns[i]

            if candid_id.sum().item() > top_n.item():
                cls_p, top_k_id = cls_p.topk(top_n, sorted=False)
                class_id = class_id[top_k_id]
                box_p = box_p[top_k_id]
                loc = loc[top_k_id]

            detections = torch.stack(
                [
                    loc[:, 0] - box_p[:, 0],
                    loc[:, 1] - box_p[:, 1],
                    loc[:, 0] + box_p[:, 2],
                    loc[:, 1] + box_p[:, 3],
                ],
                1,
            )
            # height, width = image_sizes[i]
            height, width = image_sizes

            boxlist = BoxList(detections, (int(width), int(height)), mode='xyxy')
            boxlist.fields['labels'] = class_id
            boxlist.fields['scores'] = torch.sqrt(cls_p)
            boxlist = boxlist.clip(remove_empty=False)
            boxlist = remove_small_box(boxlist, self.min_size)

            results.append(boxlist)

        return results

    def forward(self, location, cls_pred, box_pred, center_pred, image_sizes):
        boxes = []

        for loc, cls_p, box_p, center_p in zip(
            location, cls_pred, box_pred, center_pred
        ):
            boxes.append(
                self.forward_single_feature_map(
                    loc, cls_p, box_p, center_p, image_sizes
                )
            )

        boxlists = list(zip(*boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_scales(boxlists)

        return boxlists

    def select_over_scales(self, boxlists):
        results = []

        for boxlist in boxlists:
            scores = boxlist.fields['scores']
            labels = boxlist.fields['labels']
            box = boxlist.box

            result = []

            for j in range(1, self.n_class):
                id = (labels == j).nonzero().view(-1)
                score_j = scores[id]
                box_j = box[id, :].view(-1, 4)
                box_by_class = BoxList(box_j, boxlist.size, mode='xyxy')
                box_by_class.fields['scores'] = score_j
                box_by_class = boxlist_nms(box_by_class, score_j, self.nms_threshold)
                n_label = len(box_by_class)
                box_by_class.fields['labels'] = torch.full(
                    (n_label,), j, dtype=torch.int64, device=scores.device
                )
                result.append(box_by_class)

            result = cat_boxlist(result)
            n_detection = len(result)

            if n_detection > self.post_top_n > 0:
                scores = result.fields['scores']
                img_threshold, _ = torch.kthvalue(
                    scores.cpu(), n_detection - self.post_top_n + 1
                )
                keep = scores >= img_threshold.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]

            results.append(result)

        return results


class FPNTop(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        in_channels = opt.feat_channels[-1]
        self.p6 = nn.Conv2d(in_channels, opt.out_channel, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(opt.out_channel, opt.out_channel, 3, stride=2, padding=1)

        self.apply(init_conv_kaiming)

        self.use_p5 = opt.use_p5

    def forward(self, f5, p5):
        input = p5 if self.use_p5 else f5

        p6 = self.p6(input)
        p7 = self.p7(F.relu(p6))

        return p6, p7


class FPN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.inner_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()

        for i, in_channel in enumerate(opt.feat_channels, 1):
            if in_channel == 0:
                self.inner_convs.append(None)
                self.out_convs.append(None)

                continue

            inner_conv = nn.Conv2d(in_channel, opt.out_channel, 1)
            feat_conv = nn.Conv2d(opt.out_channel, opt.out_channel, 3, padding=1)

            self.inner_convs.append(inner_conv)
            self.out_convs.append(feat_conv)

        self.apply(init_conv_kaiming)

        if opt.top_blocks:
            self.top_blocks = FPNTop(opt)
        else:
            self.top_blocks = None

    def forward(self, inputs):
        inner = self.inner_convs[-1](inputs[-1])
        outs = [self.out_convs[-1](inner)]

        for feat, inner_conv, out_conv in zip(
            inputs[:-1][::-1], self.inner_convs[:-1][::-1], self.out_convs[:-1][::-1]
        ):
            if inner_conv is None:
                continue

            inner_feat = inner_conv(feat)
            _, _, H, W = inner_feat.shape
            upsample = F.upsample(inner, size=(H,W), mode='bilinear')

            inner = inner_feat + upsample
            outs.insert(0, out_conv(inner))

        if self.top_blocks is not None:
            top_outs = self.top_blocks(outs[-1], inputs[-1])
            outs.extend(top_outs)

        return outs


class FCOSHead(nn.Module):
    def __init__(self, in_channel, n_class, n_conv, prior):
        super().__init__()

        n_class = n_class - 1

        cls_tower = []
        bbox_tower = []

        for i in range(n_conv):
            cls_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            cls_tower.append(nn.GroupNorm(32, in_channel))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            bbox_tower.append(nn.GroupNorm(32, in_channel))
            bbox_tower.append(nn.ReLU())

        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)

        self.cls_pred = nn.Conv2d(in_channel, n_class, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channel, 4, 3, padding=1)
        self.center_pred = nn.Conv2d(in_channel, 1, 3, padding=1)

        self.apply(init_conv_std)

        prior_bias = -math.log((1 - prior) / prior)
        nn.init.constant_(self.cls_pred.bias, prior_bias)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

    def forward(self, input):
        logits  = []
        bboxes  = []
        centers = []

        for feat, scale in zip(input, self.scales):
            cls_out = self.cls_tower(feat)

            logits.append(self.cls_pred(cls_out))
            centers.append(self.center_pred(cls_out))

            bbox_out = self.bbox_tower(feat)
            bbox_out = torch.exp(scale(self.bbox_pred(bbox_out)))

            bboxes.append(bbox_out)

        return logits, bboxes, centers


class Topograph(nn.Module):

    def __init__(self, opt, Topograph_m = False, train = True):
        super(Topograph, self).__init__()
        
        self.fpn = FPN(opt)
        self.extractor = ResNet101(in_channel=3, pretrained=True)
        # self.extractor = ResNet50(in_channel=3, pretrained=False)
        # self.extractor = VGG16(in_channels=3)
        self.postprocessor = FCOSPostprocessor(opt)

        if Topograph_m:
            self.with_ctr = True
        self.loss = FCOSLoss(opt)

        self.head = FCOSHead(opt.out_channel, opt.n_class, opt.n_conv, opt.prior)
        self.fpn_strides = opt.fpn_strides

    def forward(self, imgs, image_sizes=None, targets=None, train=True, domain='Source'):
        b, c, h, w = imgs.shape
        features = self.extractor(imgs)
        features = self.fpn(features)
        cls_pred, box_pred, center_pred = self.head(features)
        location = self.compute_location(features)

        if train:
            if domain == 'Source':
                loss_cls, loss_box, loss_center = self.loss(
                    location, cls_pred, box_pred, center_pred, targets
                )
                losses = {
                        'loss_cls': loss_cls,
                        'loss_box': loss_box,
                        'loss_center': loss_center,
                        }

                return (features, cls_pred, box_pred, center_pred), None, losses
            elif domain == 'Target':
                return (features, cls_pred, box_pred, center_pred), None, None

        else:
            boxes = self.postprocessor(
                location, cls_pred, box_pred, center_pred, image_sizes
            )

            return boxes, cls_pred

    def compute_location(self, features):
        locations = []

        for i, feat in enumerate(features):
            _, _, height, width = feat.shape
            location_per_level = self.compute_location_per_level(
                height, width, self.fpn_strides[i], feat.device
            )
            locations.append(location_per_level)

        return locations
    
    def compute_location_per_level(self, height, width, stride, device):
        shift_x = torch.arange(
            0, width * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y = torch.arange(
            0, height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location = torch.stack((shift_x, shift_y), 1) + stride // 2

        return location
    
    def _forward_target(self, box_cls, box_regression, centerness):
        scores = []
        for i, cls_logits in enumerate(box_cls):
            if self.with_ctr:

                mask = (centerness[i].sigmoid()>0.5).float()
                scores.append((cls_logits.sigmoid() * mask).detach())
            else:
                scores.append(cls_logits.sigmoid().detach())

        return scores


if __name__ == "__main__":
    from utils.config import opt
    #from graph_matching import GModule
    model = Topograph(opt)
    #graph_model = GModule(in_channels=3, num_classes=4, device='cpu')
    input = torch.rand(4,3,256,256)
    output = model(input)
    print(output[0].shape)