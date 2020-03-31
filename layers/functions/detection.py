import torch
import torch.nn as nn
from torch.autograd import Function
from ..box_utils import decode, nms
from data import voc as cfg
import torchvision


@torch.jit.script
def batch_nms(loc_data, conf_data, prior_data):
    num_priors = prior_data.size(0)
    # scores'shape is [N, CLS]
    scores = conf_data.view(num_priors, 21)
    # bpxes's shape is [N, 4]
    boxes = decode(loc_data[0], prior_data)

    # create labels for each prediction
    boxes = boxes.view(num_priors, 1, 4).expand(num_priors, 21, 4)
    labels = torch.arange(21)
    labels = labels.view(1, 21).expand_as(scores)

    # remove predictions with the background label
    boxes = boxes[:, 1:]
    scores = scores[:, 1:]
    labels = labels[:, 1:]
    
    # batch everything, by making every class prediction be a separate instance
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    
    # # remove low scoring boxes
    inds = torch.nonzero(scores > 0.01).squeeze(1)
    boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

    output = torch.zeros(1, 6)
    if boxes.numel() == 0:
        return output
    else:
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = labels.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = nms(boxes_for_nms, scores, torch.tensor(0.45))
        #keep = torchvision.ops.boxes.batched_nms(boxes, scores, labels, 0.45)

        # keep only topk scoring predictions
        keep = keep[:200]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        labels = labels.to(torch.float32)
        labels = labels.unsqueeze(1)
        scores = scores.unsqueeze(1)
        output = torch.cat((labels, scores, boxes), 1)
        return output

class Detect(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        output = batch_nms(loc_data, conf_data, prior_data)
        return output
