import math

import torch
import torch.nn as nn
from torchvision.models import resnet18

from .preprocess import Preprocessor


class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-measure pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)

        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.sig(out)
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out


class MetaIQA(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.resnet_layer = resnet18(pretrained=False)
        self.net = BaselineModel1(1, 0.5, 1000)
        state_dict = torch.load(r'./MetaIQA/metaiqa.pth')
        self.load_state_dict(state_dict, strict=True)

        self.gpu = opt.gpu
        self.eval()
        if self.gpu:
            self.cuda()

        self.preprocessor = Preprocessor([224, 224])

    @torch.no_grad()
    def forward(self, x):
        x = self.preprocessor(x)
        if self.gpu: x = x.cuda()
        x = self.resnet_layer(x)
        x = self.net(x)
        return x

