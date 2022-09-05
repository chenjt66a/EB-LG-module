import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction


# add eb-lg module
# ----------------------------
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
# ---------------------------

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv1 = nn.Sequential(nn.Conv1d(128 * 2, 128, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(128 * 1, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)

        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv3 = nn.Sequential(nn.Conv1d(256 * 2, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(256 * 1, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        edited_l1_points = EditedByLG(l1_points, self.conv1, self.conv2)
        #l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, edited_l1_points)
        edited_l2_points = EditedByLG(l2_points, self.conv3, self.conv4)
        #l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, edited_l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn5(self.fc1(x))))
        x = self.drop2(F.relu(self.bn6(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        # return x, l3_points
        return x



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
