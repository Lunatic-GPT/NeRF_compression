import torch.nn as nn
import torch
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, L1=10, L2=4, skip=4, use_view_dirs=True):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = 3 * L1 * 2
        self.input_ch_views = 3 * L2 * 2
        self.skip = skip
        self.use_view_dirs = use_view_dirs

        self.net = nn.ModuleList([nn.Linear(self.input_ch, W)])
        for i in range(D-1):
            if i == skip:
                self.net.append(nn.Linear(W + self.input_ch, W))
            else:
                self.net.append(nn.Linear(W, W))

        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        if use_view_dirs:
            self.proj = nn.Linear(W + self.input_ch_views, W // 2)
        else:
            self.proj = nn.Linear(W, W // 2)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, input_pts, input_views=None):
        h = input_pts.clone()
        for i, _ in enumerate(self.net):
            h = F.relu(self.net[i](h))
            if i == self.skip:
                h = torch.cat([input_pts, h], -1)

        alpha = F.relu(self.alpha_linear(h))
        feature = self.feature_linear(h)

        if self.use_view_dirs:
            h = torch.cat([feature, input_views], -1)

        h = F.relu(self.proj(h))
        rgb = torch.sigmoid(self.rgb_linear(h))

        return rgb, alpha


class NeRF_Quant(nn.Module):
    def __init__(self, D=8, W=256, L1=10, L2=4, skip=4, use_view_dirs=True):
        super(NeRF, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.D = D
        self.W = W
        self.input_ch = 3 * L1 * 2
        self.input_ch_views = 3 * L2 * 2
        self.skip = skip
        self.use_view_dirs = use_view_dirs

        self.net = nn.ModuleList([nn.Linear(self.input_ch, W)])
        for i in range(D-1):
            if i == skip:
                self.net.append(nn.Linear(W + self.input_ch, W))
            else:
                self.net.append(nn.Linear(W, W))

        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        if use_view_dirs:
            self.proj = nn.Linear(W + self.input_ch_views, W // 2)
        else:
            self.proj = nn.Linear(W, W // 2)
        self.rgb_linear = nn.Linear(W // 2, 3)

        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, input_pts, input_views=None):
        h = input_pts.clone()
        h = self.quant(h)

        for i, _ in enumerate(self.net):
            h = F.relu(self.net[i](h))
            if i == self.skip:
                h = torch.cat([input_pts, h], -1)

        alpha = F.relu(self.alpha_linear(h))
        alpha = self.dequant(alpha)

        feature = self.feature_linear(h)
        if self.use_view_dirs:
            h = torch.cat([feature, input_views], -1)

        h = F.relu(self.proj(h))
        rgb = torch.sigmoid(self.rgb_linear(h))
        rgb = self.dequant(rgb)

        return rgb, alpha
