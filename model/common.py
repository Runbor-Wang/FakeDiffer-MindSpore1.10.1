import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
# from torch.nn import functional as F
import math


def freeze_weights(module):
    for param in module.parameters():
        param.requires_grad = False


def l1_regularize(module):
    reg_loss = 0.
    for key, param in module.reg_params.items():
        if "weight" in key and param.requires_grad:
            reg_loss += ms.ops.environ_add(ms.ops.abs(param))
    return reg_loss


class SeparableConv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                               groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def construct(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Cell):
    def __init__(self, in_channels, out_channels, reps, strides=1,
                 start_with_relu=True, grow_first=True, with_bn=True):
        super(Block, self).__init__()

        self.with_bn = with_bn

        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            if with_bn:
                self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if i < (reps - 1) else out_channels
            rep.append(nn.ReLU())
            rep.append(SeparableConv2d(inc, outc, 3, stride=1, padding=1))
            if with_bn:
                rep.append(nn.BatchNorm2d(outc))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU()

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.SequentialCell(*rep)

    def construct(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            if self.with_bn:
                skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


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


def ms_index_select(input_tensor, dim, index):
    def process(tensor, current_dim):
        if current_dim < dim:
            # 继续递归处理下一层
            return [process(sub_tensor, current_dim + 1) for sub_tensor in tensor]
        else:
            # 在当前层级进行索引选择
            return [tensor[i] for i in index]
    return process(input_tensor, 0)


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
    # 检查输入张量维度是否正确
    assert len(input_tensor.shape) == 4, "输入张量必须是4维"

    N, C, H, W = input_tensor.shape
    H_new, W_new = size

    # 初始化输出张量
    output_tensor = [[[[0.0 for _ in range(W_new)] for __ in range(H_new)] for ___ in range(C)] for ____ in range(N)]

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


def exp_recons_loss(recons_real, recons_fake, total):
    x, y = total
    context.set_context(device_target="GPU")
    loss_real = Tensor(0.)
    loss_fake = Tensor(0.)
    real_index = ms_where_index(1 - y)
    fake_index = ms_where_index(y)
    for real_rec in recons_real:
        if len(real_index) > 0:
            real_x = ms_index_select(x, dim=0, index=real_index)
            # real_rec = F.interpolate(real_rec.unsqueeze(0), size=x.shape[-2:], mode='bilinear', align_corners=True)
            real_rec = ms_interpolate(real_rec.unsqueeze(0), size=x.shape[-2:], mode='bilinear', align_corners=True)
            loss_real += ops.mean(ops.abs(real_rec - real_x))

    for fake_rec in recons_fake:
        if len(fake_index) > 0:
            fake_x = ms_index_select(x, dim=0, index=fake_index)
            # fake_rec = torch.index_select(r, dim=0, index=fake_index)
            # fake_rec = F.interpolate(fake_rec.unsqueeze(0), size=x.shape[-2:], mode='bilinear', align_corners=True)
            fake_rec = ms_interpolate(fake_rec.unsqueeze(0), size=x.shape[-2:], mode='bilinear', align_corners=True)
            loss_fake += ops.mean(ops.abs(fake_rec - fake_x))
    return loss_real + loss_fake


class AbstractRealFake(nn.Cell):
    """ Abstract the intermediate feature tensor of Fake. """
    def __init__(self):
        super(AbstractRealFake, self).__init__()

    def forward(self, x, y, stage):
        if stage == "reconstruction":
            real_index = ms_where_index(1 - y)
            fake_index = ms_where_index(y)
            if len(real_index) > 0:
                out = ms_index_select(x, dim=0, index=real_index)
                # print("89 out.size() :", out.size())  # torch.Size([1, 728, 19, 19])
            else:
                out = None
            if len(fake_index) > 0:
                out_fake = ms_index_select(x, dim=0, index=fake_index)
                # print("88 out_fake.size() :", out_fake.size())  # torch.Size([1, 728, 19, 19])
            else:
                out_fake = None

        elif stage == "differ":
            out = x
            out_fake = x
            # print("92 out_fake.size() :", out_fake.size())  # torch.Size([2, 728, 19, 19])
            # print("93 out.size() :", out.size())  # torch.Size([2, 728, 19, 19])
        else:
            raise ValueError(f"Error: the running stage must in ['reconstruction', 'differ'], but got {stage}.")
        return out, out_fake


class AbstractFake(nn.Cell):
    """ Abstract the intermediate feature tensor of Fake. """
    def __init__(self):
        super(AbstractFake, self).__init__()

    def construct(self, x, y):
        indices = ms_where_index(y == 1)
        return ops.Stack([x[i] for i in indices], dim=0)


class AbstractReal(nn.Cell):
    """ Abstract the intermediate feature tensor of Fake. """
    def __init__(self):
        super(AbstractReal, self).__init__()

    def forward(self, x, y):
        indices = ms_where_index(y == 0)
        return ops.Stack([x[i] for i in indices], dim=0)


class ErrorMapping(nn.Cell):
    """ Reconstruction Guided Attention. """

    def __init__(self, depth=728, drop_rate=0.2):
        super(ErrorMapping, self).__init__()
        self.depth = depth
        self.h = nn.SequentialCell(nn.Conv2d(3, depth, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(depth), nn.Conv2d(depth, depth, 1, bias=False), nn.ReLU(),)

    def forward(self, x, pred_x, embedding):
        residual_full = ops.abs(x - pred_x)
        # residual_x = F.interpolate(residual_full, size=embedding.shape[-2:], mode='bilinear', align_corners=True)
        residual_x = ms_interpolate(residual_full, size=embedding.shape[-2:], mode='bilinear', align_corners=True)
        return self.dropout(self.h(residual_x))
