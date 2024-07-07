import torch
import torch.nn as nn
import torch.nn.functional as F


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        print(x_list[1]-x_list[0])
        x = torch.cat(x_list, dim=1)
        return x


class Get_curvature(nn.Module):
    def __init__(self):
        super(Get_curvature, self).__init__()
        kernel_v1 = [[0, -1, 0],
                     [0, 0, 0],
                     [0, 1, 0]]
        kernel_h1 = [[0, 0, 0],
                     [-1, 0, 1],
                     [0, 0, 0]]
        kernel_h2 = [[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [1, 0, -2, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]
        kernel_v2 = [[0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, -2, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0]]
        kernel_w2 = [[1, 0, -1],
                     [0, 0, 0],
                     [-1, 0, 1]]
        kernel_h1 = torch.FloatTensor(kernel_h1).unsqueeze(0).unsqueeze(0)
        kernel_v1 = torch.FloatTensor(kernel_v1).unsqueeze(0).unsqueeze(0)
        kernel_v2 = torch.FloatTensor(kernel_v2).unsqueeze(0).unsqueeze(0)
        kernel_h2 = torch.FloatTensor(kernel_h2).unsqueeze(0).unsqueeze(0)
        kernel_w2 = torch.FloatTensor(kernel_w2).unsqueeze(0).unsqueeze(0)
        self.weight_h1 = nn.Parameter(data=kernel_h1, requires_grad=False)
        self.weight_v1 = nn.Parameter(data=kernel_v1, requires_grad=False)
        self.weight_v2 = nn.Parameter(data=kernel_v2, requires_grad=False)
        self.weight_h2 = nn.Parameter(data=kernel_h2, requires_grad=False)
        self.weight_w2 = nn.Parameter(data=kernel_w2, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v1, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h1, padding=1)
            x_i_v2 = F.conv2d(x_i.unsqueeze(1), self.weight_v2, padding=2)
            x_i_h2 = F.conv2d(x_i.unsqueeze(1), self.weight_h2, padding=2)
            x_i_w2 = F.conv2d(x_i.unsqueeze(1), self.weight_w2, padding=1)
            sum = torch.pow((torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2)), 3 / 2)
            fg = torch.mul(torch.pow(x_i_v, 2), x_i_v2) + 2 * torch.mul(torch.mul(x_i_v, x_i_h), x_i_w2) + torch.mul(
                torch.pow(x_i_h, 2), x_i_h2)
            fh = torch.mul(torch.pow(x_i_v, 2), x_i_h2) - 2 * torch.mul(torch.mul(x_i_v, x_i_h), x_i_w2) + torch.mul(
                torch.pow(x_i_h, 2), x_i_v2)
            x_i = torch.div(torch.abs(fg - fh), sum + 1e-10)
            x_i = torch.div(torch.abs(fh), sum + 1e-10)
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self, out_dims):
        super(FeatureEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, out_dims[0], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dims[0], out_dims[0], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(out_dims[0], out_dims[1], kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(out_dims[1], out_dims[1], kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(out_dims[1], out_dims[2], kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(out_dims[2], out_dims[2], kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(out_dims[2], out_dims[3], kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(out_dims[3], out_dims[3], kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Stage 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x1 = x

        # Stage 2
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)
        x2 = x

        # Stage 3
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool3(x)
        x3 = x

        # Stage 4
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x4 = x

        return x1, x2, x3, x4


class PMD_features(nn.Module):
    def __init__(self, out_dims):
        super(PMD_features, self).__init__()
        # self.PMD_head = Get_curvature()
        self.PMD_head = Get_gradient_nopadding()
        # self.feature_ext = FeatureEncoder(out_dims)

    def forward(self, images):
        PMD_images = self.PMD_head(images)
        # PMD_feature = self.feature_ext(PMD_images)

        return PMD_images


# class Adapter(nn.Module):
#     def __init__(self, out_dims):
#         super(Adapter, self).__init__()
#         self.PMD_head = Get_gradient_nopadding()
#         self.feature_ext = FeatureEncoder(out_dims)
#
#     def forward(self, images):
#         PMD_images = self.PMD_head(images)
#         PMD_feature = self.feature_ext(PMD_images)
#
#         return PMD_feature
