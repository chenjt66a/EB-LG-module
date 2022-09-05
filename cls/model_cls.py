import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

# Adapt
def adapt_get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx


# dgcnn
def dgcnn_get_graph_feature(x, gpu_idx, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class AdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, feat_channels):
        super(AdaptiveConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels

        self.conv0 = nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(out_channels, out_channels*in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y):
        # x: (bs, in_channels, num_points, k), y: (bs, feat_channels, num_points, k)
        batch_size, n_dims, num_points, k = x.size()

        y = self.conv0(y) # (bs, out, num_points, k)
        y = self.leaky_relu(self.bn0(y))
        y = self.conv1(y) # (bs, in*out, num_points, k)
        y = y.permute(0, 2, 3, 1).view(batch_size, num_points, k, self.out_channels, self.in_channels) # (bs, num_points, k, out, in)

        x = x.permute(0, 2, 3, 1).unsqueeze(4) # (bs, num_points, k, in_channels, 1)
        x = torch.matmul(y, x).squeeze(4) # (bs, num_points, k, out_channels)
        x = x.permute(0, 3, 1, 2).contiguous() # (bs, out_channels, num_points, k)

        x = self.bn1(x)
        x = self.leaky_relu(x)

        return x


def global_cat(keypoints_local_feature, global_feature_nd):

    batch_size, keypoints_num, feature_channels = keypoints_local_feature.shape
    global_feature_nd = global_feature_nd.view(batch_size, 1, feature_channels).contiguous()
    global_feature_nd = global_feature_nd.repeat(1, keypoints_num, 1)
    local_global_feature = torch.cat((keypoints_local_feature, global_feature_nd), dim=-1)

    return local_global_feature


def LG_feature_module(conv6, local_global_feature):
    LG_feature = local_global_feature.transpose(1, 2).contiguous()
    LG_feature = conv6(LG_feature)

    return LG_feature


def EditedByLG(x, conva, convb):
    x_hat = x.transpose(1, 2).contiguous()
    glocal_feature = torch.max(x, dim=-1, keepdim=False)[0]
    local_global_feature = global_cat(x_hat, glocal_feature)
    LG_feature = LG_feature_module(conva, local_global_feature)
    error_feature = LG_feature - x
    error_feature = convb(error_feature)
    edited_x = x + error_feature

    return edited_x


def EB_only(x, conva, convb):

    x1 = conva(x)  # 32 * 64 * 1024 --> 32 * 64 * 1024
    error_feature = x1 - x  # 32 * 64 * 1024
    error_feature = convb(error_feature)  # 32 * 64 * 1024 --> 32 * 64 * 1024
    edited_x = x + error_feature  # 32 * 64 * 1024

    return edited_x


def LG_only(x, conva):
    x_hat = x.transpose(1, 2).contiguous()  # 32 * 1024 * 64
    glocal_feature = torch.max(x, dim=-1, keepdim=False)[0]  # 32 * 64
    local_global_feature = global_cat(x_hat, glocal_feature)  # 32 * 1024 * 128
    edited_x = LG_feature_module(conva, local_global_feature)  # 32 * 64 *  1024

    return edited_x


# AdaptConv+EB-LG
class AdaptEBLG(nn.Module):
    def __init__(self, args, output_channels=40):
        super(AdaptEBLG, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(128)
        self.bn10 = nn.BatchNorm2d(256)
        self.bn11 = nn.BatchNorm1d(256)
        self.bn12 = nn.BatchNorm1d(256)
        self.bn13 = nn.BatchNorm1d(args.emb_dims)

        self.adapt_conv1 = AdaptiveConv(6, 64, 6)

        self.conv2 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64 * 1, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.adapt_conv4 = AdaptiveConv(6, 64, 64 * 2)

        self.conv5 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(64 * 1, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(128 * 1, 128, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv1d(256 * 2, 256, kernel_size=1, bias=False),
                                   self.bn11,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv12 = nn.Sequential(nn.Conv1d(256 * 1, 256, kernel_size=1, bias=False),
                                   self.bn12,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv13 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn13,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn14 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn15 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)


    def forward(self, x):
        batch_size = x.size(0)
        points = x  # 32 * 3 1024

        x, idx = adapt_get_graph_feature(x, k=self.k)  # 32 * 6 * 1024 * 20
        p, _ = adapt_get_graph_feature(points, k=self.k, idx=idx)  # 32 * 6 * 1024 * 20
        x = self.adapt_conv1(p, x)  # 32 * 64 * 1024 * 20
        x1 = x.max(dim=-1, keepdim=False)[0]  # 32 * 64 * 1024

        edited_x1 = EditedByLG(x1, self.conv2, self.conv3)  # 32 * 64 * 1024

        x, idx = adapt_get_graph_feature(x1, k=self.k)  # 32 * 128 * 1024 * 20
        p, _ = adapt_get_graph_feature(points, k=self.k, idx=idx)  # 32 * 6 * 1024 * 20
        x = self.adapt_conv4(p, x)  # 32 * 64 * 1024 * 20
        x2 = x.max(dim=-1, keepdim=False)[0]  # 32 * 64 * 1024

        edited_x2 = EditedByLG(x2, self.conv5, self.conv6)  # 32 * 64 * 1024

        x, _ = adapt_get_graph_feature(x2, k=self.k)  # 32 * 128 * 1024 * 20
        x = self.conv7(x)  # 32 * 128 * 1024 * 20
        x3 = x.max(dim=-1, keepdim=False)[0]  # 32 * 128 * 1024

        edited_x3 = EditedByLG(x3, self.conv8, self.conv9)  # 32 * 128 * 1024

        x, _ = adapt_get_graph_feature(x3, k=self.k)  # 32 * 256 * 1024 * 20
        x = self.conv10(x)  # 32 * 256 * 1024 * 20
        x4 = x.max(dim=-1, keepdim=False)[0]  # 32 * 256 * 1024

        edited_x4 = EditedByLG(x4, self.conv11, self.conv12)  # 32 * 256 * 1024

        x = torch.cat((edited_x1, edited_x2, edited_x3, edited_x4), dim=1)  # 32 * 512 * 1024

        x = self.conv13(x)  # 32 * 1024f * 1024p
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # 32 * 1024
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # 32 * 1024
        x = torch.cat((x1, x2), 1)  # 32 * 2048

        x = F.leaky_relu(self.bn14(self.linear1(x)), negative_slope=0.2)  # 32 * 512
        x = self.dp1(x)
        x = F.leaky_relu(self.bn15(self.linear2(x)), negative_slope=0.2)  # 32 * 256
        x = self.dp2(x)
        x = self.linear3(x)
        return x


# DGCNN+EB-LG
class DgcnnEBLG(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DgcnnEBLG, self).__init__()
        self.args = args
        self.k = args.k
        self.gpu_idx = args.gpu_idx
        self.num_points = args.num_points
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(64)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(128)
        self.bn10 = nn.BatchNorm2d(256)
        self.bn11 = nn.BatchNorm1d(256)
        self.bn12 = nn.BatchNorm1d(256)
        self.bn13 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64 * 1, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(64 * 1, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(128 * 1, 128, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv2d(256 * 1, 256, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv1d(256 * 2, 256, kernel_size=1, bias=False),
                                   self.bn11,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv12 = nn.Sequential(nn.Conv1d(256 * 1, 256, kernel_size=1, bias=False),
                                   self.bn12,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv13 = nn.Sequential(nn.Conv1d(256 * 2, 1024, kernel_size=1, bias=False),
                                   self.bn13,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn14 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn15 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):

        batch_size = x.size(0)  # 32 * 3 * 1024
        x = dgcnn_get_graph_feature(x, self.gpu_idx, k=self.k, idx=None)  # 32 * 6 * 1024 * 20
        x = self.conv1(x)  # 32 * 64 * 1024 * 20
        x1 = x.max(dim=-1, keepdim=False)[0]  # 32 * 64 * 1024

        edited_x1 = EditedByLG(x1, self.conv2, self.conv3)  # 32 * 64 * 1024

        x = dgcnn_get_graph_feature(x1, self.gpu_idx, k=self.k, idx=None)  # 32 * 128 * 1024 * 20
        x = self.conv4(x)  # 32 * 64 * 1024 * 20
        x2 = x.max(dim=-1, keepdim=False)[0]  # 32 * 64 * 1024

        edited_x2 = EditedByLG(x2, self.conv5, self.conv6)  # 32 * 64 * 1024

        x = dgcnn_get_graph_feature(x2, self.gpu_idx, k=self.k, idx=None)  # 32 * 128 * 1024 * 20
        x = self.conv7(x)  # 32 * 128 * 1024 * 20
        x3 = x.max(dim=-1, keepdim=False)[0]  # 32 * 128 * 1024

        edited_x3 = EditedByLG(x3, self.conv8, self.conv9)  # 32 * 128 * 1024

        x = dgcnn_get_graph_feature(x3, self.gpu_idx, k=self.k, idx=None)  # 32 * 128 * 1024 * 20
        x = self.conv10(x)  # 32 * 256 * 1024 * 20
        x4 = x.max(dim=-1, keepdim=False)[0]  # 32 * 256 * 1024

        edited_x4 = EditedByLG(x4, self.conv11, self.conv12)  # 32 * 256 * 1024

        x = torch.cat((edited_x1, edited_x2, edited_x3, edited_x4), dim=1)  # 32 * 64+64+128+256=512f * 1024p

        x = self.conv13(x)  # 32 * 512f-->1024f * 1024

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # 32 * 1024
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # 32 * 1024
        x = torch.cat((x1, x2), 1)  # 32 * 2048

        x = F.leaky_relu(self.bn14(self.linear1(x)), negative_slope=0.2)  # 32 * 512
        x = self.dp1(x)
        x = F.leaky_relu(self.bn15(self.linear2(x)), negative_slope=0.2)  # 32 * 256
        x = self.dp2(x)
        x = self.linear3(x)  # 32 * output_channels

        return x

