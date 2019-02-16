# -*- coding: utf-8 -*-
"""
Multimodal Attentional Translation Embeddings.

Model of our 2019 paper:
"Deeply Supervised Multimodal Attentional Translation Embeddings
for Visual Relationship Detection",
Authors:
Gkanatsios N., Pitsikalis V., Koutras P., Zlatintsi A., Maragos P..

Code by: N. Gkanatsios
"""

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
import yaml

from src.utils.train_test_utils import VRDTrainTester

with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f)


class MATransE(nn.Module):
    """MATransE main."""

    def __init__(self, use_cuda=False, mode='train'):
        """Initialize model."""
        super().__init__()
        self.p_branch = PredicateBranch(use_cuda=use_cuda)
        self.os_branch = ObjectSubjectBranch(use_cuda=use_cuda)
        self.fc_fusion = nn.Sequential(
            nn.Linear(140, 100), nn.ReLU(), nn.Linear(100, 70))
        self.softmax = nn.Softmax(dim=1)
        self.mode = mode

    def forward(self, subj_feats, pred_feats, obj_feats, masks,
                subj_embs, obj_embs):
        """Forward pass."""
        pred_scores = self.p_branch(pred_feats, masks, subj_embs, obj_embs)
        so_scores = self.os_branch(
            subj_feats, obj_feats, subj_embs, obj_embs, masks)
        scores = self.fc_fusion(torch.cat((pred_scores, so_scores), dim=1))
        if self.mode == 'test':  # scores across pairs are compared in R_70
            scores = self.softmax(scores)
        return scores, pred_scores, so_scores


class PredicateBranch(nn.Module):
    """
    Predicate Branch.

    pred. features -> CONV -> RELU -> Att. Pool -> CONV1D -> RELU ->
    -> Att. Clsfr -> out
    """

    def __init__(self, use_cuda=False):
        """Initialize model."""
        super().__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(2048, 256, 1), nn.ReLU())
        self.attention_layer = AttentionLayer()
        self.pooling_weights = AttentionalWeights(feature_dim=256)
        self.attentional_pooling = AttentionalPoolingLayer()
        self.conv_2 = nn.Sequential(nn.Conv1d(256, 128, 1), nn.ReLU())
        self.classifier_weights = AttentionalWeights(feature_dim=128)
        _bias = (torch.rand(1, 70).cuda() if use_cuda else torch.rand(1, 70))
        self.bias = nn.Parameter(_bias)

    def forward(self, pred_feats, masks, subj_embs, obj_embs):
        """Forward pass."""
        attention = self.attention_layer(subj_embs, obj_embs, masks)
        pred_feats = self.attentional_pooling(
            self.conv_1(pred_feats),
            self.pooling_weights(attention)
        )
        pred_feats = self.conv_2(pred_feats)
        return (
            torch.sum(pred_feats * self.classifier_weights(attention), dim=1)
            + self.bias
        )


class ObjectSubjectBranch(nn.Module):
    """
    Object-Subject Branch.

    obj. features  -> FC -> RELU -> FC -> RELU -> |
                                                  - -> Att. Clsfr -> out
    subj. features -> FC -> RELU -> FC -> RELU -> |
    """

    def __init__(self, use_cuda=CONFIG['use_cuda']):
        """Initialize model."""
        super().__init__()
        self.fc_subj = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.fc_obj = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.attention_layer = AttentionLayer()
        self.classifier_weights = AttentionalWeights(feature_dim=128)
        _bias = (torch.rand(1, 70).cuda() if use_cuda else torch.rand(1, 70))
        self.bias = nn.Parameter(_bias)

    def forward(self, subj_feats, obj_feats, subj_embs, obj_embs, masks):
        """Forward pass, return output scores."""
        attention = self.attention_layer(subj_embs, obj_embs, masks)
        os_feats = self.fc_obj(obj_feats) - self.fc_subj(subj_feats)
        return (
            torch.sum(
                os_feats.unsqueeze(-1) * self.classifier_weights(attention),
                dim=1)
            + self.bias
        )


class AttentionalWeights(nn.Module):
    """Compute weights based on spatio-linguistic attention."""

    def __init__(self, feature_dim, num_classes=70):
        """Initialize model."""
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feature_dim
        self.att_fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, feature_dim * num_classes), nn.ReLU()
        )

    def forward(self, attention):
        """Forward pass."""
        return self.att_fc(attention).view(-1, self.feat_dim, self.num_classes)


class AttentionLayer(nn.Module):
    """Drive attention using language and/or masks."""

    def __init__(self):
        """Initialize model."""
        super().__init__()
        self.fc_subject = nn.Sequential(nn.Linear(300, 128), nn.ReLU())
        self.fc_object = nn.Sequential(nn.Linear(300, 128), nn.ReLU())
        self.fc_embs = nn.Sequential(nn.Linear(256, 64), nn.ReLU())
        self.mask_net = nn.Sequential(
            nn.Conv2d(2, 96, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(96, 128, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(128, 64, 8), nn.ReLU()
        )

    def forward(self, subj_embs, obj_embs, masks):
        """Forward pass."""
        embeddings = torch.cat(
            (self.fc_subject(subj_embs), self.fc_object(obj_embs)),
            dim=1)
        return torch.cat((
            self.fc_embs(embeddings),
            self.mask_net(masks).view(masks.shape[0], -1)
        ), dim=1)


class AttentionalPoolingLayer(nn.Module):
    """Attentional Pooling layer."""

    def __init__(self):
        """Initialize model."""
        super().__init__()
        self.register_buffer('const', torch.FloatTensor([0.0001]))
        self.softplus = nn.Softplus()

    def forward(self, features, weights):
        """Forward pass."""
        features = features.unsqueeze(-1)  # (bs, 256, 7, 7, 1)
        weights = weights.unsqueeze(2).unsqueeze(2)  # (bs, 256, 1, 1, 70)
        att_num = (
            self.softplus(torch.sum(features * weights, dim=1))
            + self.const
        )
        att_denom = torch.sum(torch.sum(att_num, dim=2), dim=1)
        attention_map = (
            att_num  # (bs, 7, 7, 70)
            / att_denom.unsqueeze(1).unsqueeze(2)  # (bs, 1, 1, 70)
        )  # (bs, 7, 7, 70)
        return torch.sum(
            torch.sum(attention_map.unsqueeze(1) * features, dim=3),
            dim=2
        )


class TrainTester(VRDTrainTester):
    """Extends VRDTrainTester."""

    def __init__(self, net, net_name, use_cuda=CONFIG['use_cuda']):
        """Initialize instance."""
        super().__init__(net, net_name, use_cuda)

    def _compute_loss(self, data_loader, epoch, batch_start):
        """Compute loss for current batch."""
        scores, pred_scores, so_scores = self.net(
            data_loader.get_subject_boxes_pool5_features(epoch, batch_start),
            data_loader.get_union_boxes_conv5_features(epoch, batch_start),
            data_loader.get_object_boxes_pool5_features(epoch, batch_start),
            data_loader.get_masks(epoch, batch_start),
            data_loader.get_subject_embeddings(epoch, batch_start),
            data_loader.get_object_embeddings(epoch, batch_start)
        )
        targets = data_loader.get_targets(epoch, batch_start)
        loss = (
            1.5 * self.criterion(scores, targets)
            + self.criterion(pred_scores, targets)
            + self.criterion(so_scores, targets)
            + sum(0.01 * param.norm(2) for param in self.net.parameters()))
        return loss

    def _net_outputs(self, data_loader, epoch, batch_start):
        """Get network outputs for current batch."""
        return self.net(
            data_loader.get_subject_boxes_pool5_features(epoch, batch_start),
            data_loader.get_union_boxes_conv5_features(epoch, batch_start),
            data_loader.get_object_boxes_pool5_features(epoch, batch_start),
            data_loader.get_masks(epoch, batch_start),
            data_loader.get_subject_embeddings(epoch, batch_start),
            data_loader.get_object_embeddings(epoch, batch_start)
        )[0]


def train_test():
    """Train and test a net."""
    net = MATransE(use_cuda=CONFIG['use_cuda'])
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    train_tester = TrainTester(
        net,
        'MATransE_SLAM_POS_DS',
        use_cuda=CONFIG['use_cuda']
    )
    train_tester.train(
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        scheduler=MultiStepLR(optimizer, [5, 9]),
        epochs=10,
        batch_size=32,
        val_batch_size=1000,
        loss_sampling_period=450
    )
    train_tester.net.mode = 'test'
    train_tester.test(batch_size=1000, test_mode='relationship')

if __name__ == "__main__":
    train_test()
