from functools import partial
from model.common import SeparableConv2d, Block
from model.common import AbstractFake, AbstractReal, AbstractRealFake

import mindspore.nn as nn
from mindspore.common.initializer import HeNormal
import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor
import math


def ms_where_index(condition):
    condition_shape = condition.shape
    condition_dims = len(condition_shape)

    # 生成所有可能的索引组合
    condition_index = list()

    def generate_indices(shape):
        if not shape:
            return [[]]
        current_dim = shape[0]
        sub_indices = generate_indices(shape[1:])
        for i in range(current_dim):
            for sub in sub_indices:
                condition_index.append([i] + sub)
        return condition_index

    def get_element(arr, indices):
        for idx in indices:
            arr = arr[idx]
        return arr

    all_indices = generate_indices(condition_shape)
    result = [[] for _ in range(condition_dims)]

    for coords in all_indices:
        try:
            elem = get_element(condition, coords)
        except (IndexError, TypeError):
            continue
        if elem is True:
            for i in range(condition_dims):
                result[i].append(coords[i])

    return tuple(result)


class SeparableConv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride,
            pad_mode='pad', padding=padding, group=in_channels,
            weight_init=HeNormal(), has_bias=False)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1,
            weight_init=HeNormal(), has_bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MSUpsamplingNearest2d(nn.Cell):
    def __init__(self, scale_factor=2):
        super(MSUpsamplingNearest2d, self).__init__()
        self.scale_factor = scale_factor

    def ms_interpolate(input_tensor, size, mode='bilinear', align_corners=True):
        """
        自定义双线性插值函数
        Args:
            input_tensor: 输入张量 [N, C, H, W]
            size: 目标尺寸 (H_new, W_new)
            mode: 插值模式，默认为 'bilinear'
            align_corners: 是否对齐角落，默认为 True
        Returns:
            插值后的张量 [N, C, H_new, W_new]
        """
        # 检查输入张量是否为4维
        assert len(input_tensor.shape) == 4, "输入张量必须是4维"

        N, C, H, W = input_tensor.shape
        H_new, W_new = size

        # 初始化输出张量
        output_tensor = [[[[0.0 for _ in range(W_new)] for __ in range(H_new)] for ___ in range(C)] for ____ in
                         range(N)]

        for n in range(N):
            for c in range(C):
                for h_new in range(H_new):
                    for w_new in range(W_new):
                        # 计算对应位置
                        if align_corners:
                            if H == 1 or H_new == 1:
                                h = 0.0
                            else:
                                h = (h_new * (H - 1)) / (H_new - 1)

                            if W == 1 or W_new == 1:
                                w = 0.0
                            else:
                                w = (w_new * (W - 1)) / (W_new - 1)
                        else:
                            if H == 0 or H_new == 0:
                                h = 0.0
                            else:
                                h = ((h_new + 0.5) * H) / H_new - 0.5

                            if W == 0 or W_new == 0:
                                w = 0.0
                            else:
                                w = ((w_new + 0.5) * W) / W_new - 0.5

                                # 处理边界情况
                        if h < 0:
                            h = 0.0
                        elif h >= H - 1e-8:  # 处理浮点数精度问题
                            h = H - 1e-8

                        if w < 0:
                            w = 0.0
                        elif w >= W - 1e-8:
                            w = W - 1e-8

                            # 确定四个最近邻点的位置
                        h_low = int(math.floor(h))
                        h_high = h_low + 1

                        w_low = int(math.floor(w))
                        w_high = w_low + 1

                        # 确保索引不超过范围
                        if h_high >= H:
                            h_high -= 1
                            if h_high < h_low:
                                h_high = h_low

                        if w_high >= W:
                            w_high -= 1
                            if w_high < w_low:
                                w_high = w_low

                                # 计算插值权重
                        alpha = h - h_low
                        beta = w - w_low

                        # 获取四个点的值
                        A_val = input_tensor[n][c][h_low][w_low]
                        B_val = input_tensor[n][c][h_low][w_high]
                        C_val = input_tensor[n][c][h_high][w_low]
                        D_val = input_tensor[n][c][h_high][w_high]

                        # 双线性插值计算
                        interpolated_val = (A_val * (1 - alpha) * (1 - beta) + B_val * (1 - alpha) * beta +
                                            C_val * alpha * (1 - beta) + D_val * alpha * beta)

                        # 存储结果
                        output_tensor[n][c][h_new][w_new] = interpolated_val

        return output_tensor

    def construct(self, x):
        return self.ms_interpolate(x)


class ReconstructionXception(nn.Cell):
    """ End-to-End Reconstruction-Classification Learning for Face Forgery Detection """
    def __init__(self, drop_rate=0.2):
        super(ReconstructionXception, self).__init__()
        # Entry Flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, padding=0, pad_mode='valid', weight_init=HeNormal())
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, pad_mode='pad', weight_init=HeNormal())
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()

        # Entry flow blocks
        self.block1 = self._make_entry_flow_block(64, 128, stride=2)
        self.block2 = self._make_entry_flow_block(128, 256, stride=2)
        self.block3 = self._make_entry_flow_block(256, 728, stride=2)

        self.dropout = nn.Dropout(drop_rate)

        # Middle flow (repeat 8 times)
        self.middle_flow = nn.SequentialCell(*[self._make_middle_flow_block() for _ in range(8)])

        # Exit flow
        self.exit_flow = self._make_exit_flow()

        # Final layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # ==================================   =============================================
        self.abstract_fake = AbstractFake()
        self.abstract_real = AbstractReal()
        self.abstract_real_fake = AbstractRealFake()

        self.decoder_r1 = nn.SequentialCell(MSUpsamplingNearest2d(scale_factor=2),
                                            SeparableConv2d(728, 256, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(256), nn.ReLU())
        self.decoder_f1 = nn.SequentialCell(MSUpsamplingNearest2d(scale_factor=2),
                                            SeparableConv2d(728, 256, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(256), nn.ReLU())
        self.decoder_r2 = Block(256, 256, 3, 1)
        self.decoder_f2 = Block(256, 256, 3, 1)
        self.decoder_r3 = nn.SequentialCell(MSUpsamplingNearest2d(scale_factor=2),
                                            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(128), nn.ReLU())
        self.decoder_f3 = nn.SequentialCell(MSUpsamplingNearest2d(scale_factor=2),
                                            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(128), nn.ReLU())
        self.decoder_r4 = Block(128, 128, 3, 1)
        self.decoder_f4 = Block(128, 128, 3, 1)
        self.decoder_r5 = nn.SequentialCell(MSUpsamplingNearest2d(scale_factor=2),
                                            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(64), nn.ReLU())
        self.decoder_f5 = nn.SequentialCell(MSUpsamplingNearest2d(scale_factor=2),
                                            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(64), nn.ReLU())
        self.decoder_r6 = nn.SequentialCell(nn.Conv2d(64, 3, 1, 1, bias=False), nn.Tanh())
        self.decoder_f6 = nn.SequentialCell(nn.Conv2d(64, 3, 1, 1, bias=False), nn.Tanh())

    def _make_entry_flow_block(self, in_channels, out_channels, stride):
        residual = nn.SequentialCell(nn.Conv2d(in_channels, out_channels, 1, stride=stride,
                                               pad_mode='valid', weight_init=HeNormal()), nn.BatchNorm2d(out_channels))

        main = nn.SequentialCell(SeparableConv2d(in_channels, in_channels, stride=1),
                                 nn.ReLU(), SeparableConv2d(in_channels, out_channels, stride=1),
                                 nn.MaxPool2d(kernel_size=3, stride=stride, pad_mode='same'))
        return nn.SequentialCell(main, residual)

    def _make_middle_flow_block(self):
        return nn.SequentialCell(nn.ReLU(), SeparableConv2d(728, 728, stride=1), nn.ReLU(),
                                 SeparableConv2d(728, 728, stride=1), nn.ReLU(), SeparableConv2d(728, 728, stride=1))

    def _make_exit_flow(self):
        residual = nn.SequentialCell(nn.Conv2d(728, 1024, 1, stride=2, pad_mode='valid', weight_init=HeNormal()),
                                     nn.BatchNorm2d(1024))

        main = nn.SequentialCell(nn.ReLU(), SeparableConv2d(728, 728), nn.ReLU(), SeparableConv2d(728, 1024),
                                 nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same'))

        exit_flow = nn.SequentialCell(nn.SequentialCell(main, residual), SeparableConv2d(1024, 1536), nn.ReLU(),
                                      SeparableConv2d(1536, 2048), nn.ReLU())
        return exit_flow

    def _xception_encoder(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)[0] + self.block1(x)[1]
        x = self.block2(x)[0] + self.block2(x)[1]
        x = self.block3(x)[0] + self.block3(x)[1]

        # Middle flow
        x = self.middle_flow(x)

        # Exit flow
        x = self.exit_flow(x)

        # Final pooling
        x = self.avg_pool(x)
        return self.flatten(x)

    @staticmethod
    def add_white_noise(tensor, mean=0., std=1e-6):
        rand = ops.rand([tensor.shape[0], 1, 1, 1])
        rand = ms_where_index(rand > 0.5, 1., 0.).to(tensor.device)
        white_noise = ops.normal(mean, std, size=tensor.shape, device=tensor.device)
        noise_t = tensor + white_noise * rand
        noise_t = ops.clamp(noise_t, -1., 1.)
        return noise_t

    def forward(self, x, y, stage):  # torch.Size([2, 3, 299, 299])
        # clear the loss inputs
        noise_x = self.add_white_noise(x) if self.training else x
        out = self.conv1(noise_x)
        out = self.bn1(out)
        out = self.act1(out)
        # print("68 out.size() :", out.size())  # torch.Size([2, 32, 149, 149])
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        # print("72 out.size() :", out.size())  # torch.Size([2, 64, 147, 147])
        out = self.block1(out)
        # print("74 out.size() :", out.size())  # torch.Size([2, 128, 74, 74])
        out = self.block2(out)
        # print("76 out.size() :", out.size())  # torch.Size([2, 256, 37, 37])
        embedding = self.encoder.block3(out)
        # print("78 out.size() :", out.size())  # torch.Size([2, 728, 19, 19])
        # embedding = self.encoder.block4(out)
        # print("80 out.size() :", out.size())  # torch.Size([2, 728, 19, 19])

        out_deep = self.dropout(embedding)
        # print("83 out.size() :", out.size())  # torch.Size([2, 728, 19, 19])

        out, out_fake = self.abstract_real_fake(out_deep, y, stage)


        if out is not None:
            out = self.decoder_r1(out)
            # print("92 out.size() :", out.size())  # torch.Size([1, 256, 38, 38])
            out = self.decoder_r2(out)
            # print("95 out_d2.size() :", out.size())  # torch.Size([1, 256, 38, 38])
            out = self.decoder_r3(out)
            # print("99 out.size() :", out.size())  # torch.Size([1, 128, 76, 76])
            out = self.decoder_r4(out)
            # print("102 out_d4.size() :", out.size())  # torch.Size([1, 128, 76, 76])
            out = self.decoder_r5(out)
            # print("106 out.size() :", out.size())  # torch.Size([1, 64, 152, 152])
            pred = self.decoder_r6(out)
            # print("109 pred.size() :", pred.size())  # torch.Size([1, 3, 152, 152])
            recons_x = MSUpsamplingNearest2d()(pred, size=x.shape[-2:], mode='bilinear', align_corners=True)
            # print("113 recons_x.size() :", recons_x.size())  # torch.Size([1, 3, 299, 299])
        else:
            recons_x = out

        if out_fake is not None:
            out_fake = self.decoder_f1(out_fake)
            # print("92 out_fake.size() :", out_fake.size())  # torch.Size([1, 256, 38, 38])
            out_fake = self.decoder_f2(out_fake)
            # print("92 out_fake.size() :", out_fake.size())  # torch.Size([1, 256, 38, 38])
            out_fake = self.decoder_f3(out_fake)
            # print("92 out_fake.size() :", out_fake.size())  # torch.Size([1, 256, 38, 38])
            out_fake = self.decoder_f4(out_fake)
            # print("92 out_fake.size() :", out_fake.size())  # torch.Size([1, 256, 38, 38])
            out_fake = self.decoder_f5(out_fake)
            # print("92 out_fake.size() :", out_fake.size())  # torch.Size([1, 256, 38, 38])
            pred_fake = self.decoder_f6(out_fake)
            # print("92 out_fake.size() :", out_fake.size())  # torch.Size([1, 256, 38, 38])
            recons_x_fake = MSUpsamplingNearest2d()(pred_fake, size=x.shape[-2:], mode='bilinear', align_corners=True)
        else:
            recons_x_fake = out_fake
        return recons_x, recons_x_fake, out_deep
