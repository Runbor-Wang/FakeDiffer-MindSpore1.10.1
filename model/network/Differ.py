import mindspore.nn as nn
import mindspore.ops as ops
from functools import reduce


class Differ(nn.Cell):
    """ End-to-End Reconstruction-Classification Learning for Face Forgery Detection """

    def __init__(self, num_classes, encoder, drop_rate=0.2):
        super(Differ, self).__init__()
        self.encoder = encoder
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(2048, self.num_classes)
        # self.fc = nn.Conv2d(2048, self.num_classes, 1, 1, bias=False)

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.SequentialCell(nn.Conv2d(728, 91, 1, bias=False), nn.BatchNorm2d(91), nn.ReLU())
        self.fc2 = nn.Conv2d(91, 728 * 3, 1, 1, bias=False)
        # self.fc = nn.Conv2d(2048, self.num_classes, 1, 1, bias=False)
        self.softmax = nn.Softmax()

    def _abstract(self, x):
        out = self.encoder.conv1(x)
        out = self.encoder.bn1(out)
        out = self.encoder.act1(out)
        # print("30 out.size() :", out.size())  # torch.Size([2, 32, 149, 149])
        out = self.encoder.conv2(out)
        out = self.encoder.bn2(out)
        out = self.encoder.act2(out)
        # print("34 out.size() :", out.size())  # torch.Size([2, 64, 147, 147])
        out = self.encoder.block1(out)
        # print("36 out.size() :", out.size())  # torch.Size([2, 128, 74, 74])
        out = self.encoder.block2(out)
        # print("38 out.size() :", out.size())  # torch.Size([2, 256, 37, 37])
        # print("39 self.encoder.block3(out) :", self.encoder.block3(out).size())  # torch.Size([2, 728, 19, 19])
        return self.encoder.block3(out)

    def _srf_abstrct(self, x, y, out_deep):
        batch_size = x.size(0)
        sum_x_y = x + y + out_deep
        s = self.global_pool(sum_x_y)
        # print("46 s.size() :", s.size())  # torch.Size([32, 728, 1, 1])
        z = self.fc1(s)  # S->Z   # [batch_size,d,1,1]
        # print("48 z.size() :", z.size())  # torch.Size([32, 91, 1, 1])
        a_b_c = self.fc2(z)  # Z->a，b
        # print("50 x_y.size() :", a_b.size())  # torch.Size([32, 1456, 1, 1])
        a_b_c = a_b_c.reshape(batch_size, 3, 728, -1)  # 调整形状，变为 两个全连接层的值[batch_size,M,out_channels,1]
        # print("52 x_y.size() :", a_b.size())  # torch.Size([32, 2, 728, 1])
        a_b_c = self.softmax(a_b_c)  # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]
        # the part of selection
        a_b_c = list(a_b_c.chunk(3, dim=1))  # split to a and b
        # print("56 a_b[0].size() :", a_b[0].size())  # torch.Size([32, 1, 728, 1])
        # print("57 a_b[1].size() :", a_b[1].size())  # torch.Size([32, 1, 728, 1])
        a_b_c = list(map(lambda m: m.reshape(batch_size, 728, 1, 1), a_b_c))
        return reduce(lambda m, n: m + n, list(map(lambda m, n: m * n, [x, y, out_deep], a_b_c)))

    def construct(self, x, recons_real, recons_fake, out_deep):  # torch.Size([2, 3, 299, 299])
        residual_real = ops.abs(x - recons_real)
        residual_fake = ops.abs(x - recons_fake)
        # print("64 residual_real.size() :", residual_real.size())  # torch.Size([32, 3, 299, 299])
        # print("65 residual_fake.size() :", residual_fake.size())  # torch.Size([32, 3, 299, 299])

        out_real = self._abstract(residual_real)
        out_fake = self._abstract(residual_fake)
        # print("69 residual_real.size() :", out_real.size())  # torch.Size([32, 728, 19, 19])
        # print("70 residual_fake.size() :", out_fake.size())  # torch.Size([32, 728, 19, 19])

        s_real_fake_deep = self._srf_abstrct(out_real, out_fake, out_deep)
        # print("73 s_real_fake.size() :", s_real_fake.size())  # torch.Size([32, 728, 19, 19])

        embedding = self.encoder.block9(s_real_fake_deep)
        embedding = self.encoder.block10(embedding)
        embedding = self.encoder.block11(embedding)
        embedding = self.encoder.block12(embedding)
        # print("79 embedding.size() :", embedding.size())  # torch.Size([2, 1024, 10, 10])

        embedding = self.encoder.conv3(embedding)
        embedding = self.encoder.bn3(embedding)
        embedding = self.encoder.act3(embedding)
        embedding = self.encoder.conv4(embedding)
        embedding = self.encoder.bn4(embedding)
        embedding = self.encoder.act4(embedding)
        # print("88 embedding.size() :", embedding.size())  # torch.Size([32, 2048, 10, 10])

        embedding = self.global_pool(embedding).squeeze()
        # print("91 embedding.size() :", embedding.size())  # torch.Size([2, 2048])

        embedding = self.dropout(embedding)
        # print("94 out.size() :", out.size())  # torch.Size([2, 2048])
        # print("95 self.fc(out) :", self.fc(out))  # tensor([[-0.1641], [-0.2408]])
        # print("96 self.fc(out).size() :", self.fc(out).size())  # torch.Size([2, 1])
        return self.fc(embedding)
