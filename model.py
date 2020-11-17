import torch.nn as nn
import torch.nn.functional as F
import torch

#to extract lowlevel features
class lowLevelFeatNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(lowLevelFeatNet, self).__init__()
        # conv1 size: 1x64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        #nn.init.xavier_normal_(self.conv1.weight) #xavier init
        self.bn1 = norm_layer(64)
        #conv2 size: 64x128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_layer(128)
        #conv3 size: 128x128
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = norm_layer(128)
        #conv4 size: 128x256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = norm_layer(256)
        #conv5 size: 256x256
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = norm_layer(256)
        #conv6 size: 256x512
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = norm_layer(512)

    def forward(self, x1, x2):
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = F.relu(self.bn4(self.conv4(x1)))
        x1 = F.relu(self.bn5(self.conv5(x1)))
        x1 = F.relu(self.bn6(self.conv6(x1)))
        if self.training:
            x2 = x1.clone()
        else:
            x2 = F.relu(self.bn1(self.conv1(x2)))
            x2 = F.relu(self.bn2(self.conv2(x2)))
            x2 = F.relu(self.bn3(self.conv3(x2)))
            x2 = F.relu(self.bn4(self.conv4(x2)))
            x2 = F.relu(self.bn5(self.conv5(x2)))
            x2 = F.relu(self.bn6(self.conv6(x2)))
        return x1, x2


#to extract midelevel features
class midLevelFeatNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(midLevelFeatNet, self).__init__()
        #conv1 size: 512x512
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(512)
        #conv1 size: 512x256
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_layer(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

#to extract global features
class globalFeatNet(nn.Module):
    def __init__(self, norm_layer_1d=nn.BatchNorm1d, norm_layer_2d=nn.BatchNorm2d):
        super(globalFeatNet, self).__init__()
        #conv1 size: 512x512
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = norm_layer_2d(512)
        #conv2 size: 512x512
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_layer_2d(512)
        #conv3 size: 512x512
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn3 = norm_layer_2d(512)
        #conv4 size: 512x512
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = norm_layer_2d(512)
        #linear transform 25088x1024
        self.fc1 = nn.Linear(25088, 1024)
        self.bn5 = norm_layer_1d(1024)
        #linear transform 1024x512
        self.fc2 = nn.Linear(1024, 512)
        self.bn6 = norm_layer_1d(512)
        #linear transform 512x256
        self.fc3 = nn.Linear(512, 256)
        self.bn7 = norm_layer_1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, 25088)
        print(x.shape)
        x = F.relu(self.bn5(self.fc1(x)))
        print(x.shape)
        output_512 = F.relu(self.bn6(self.fc2(x)))
        output_256 = F.relu(self.bn7(self.fc3(output_512)))
        return output_512, output_256

#classifier to detect objects
class classificationNet(nn.Module):
    def __init__(self):
        super(classificationNet, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 205)
        self.bn2 = nn.BatchNorm1d(205)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.log_softmax(self.bn2(self.fc2(x)))
        return x

#regressor to generate mean chrominance
class colorizationNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(colorizationNet, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        #conv1 size: 256x128
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_layer(128)
        #conv2 size: 128x64
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(64)
        #conv3 size: 64x64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = norm_layer(64)
        #conv4 size: 64x32
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = norm_layer(32)
        #conv5 size: 32x2
        self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, mid_input, global_input):
        w = mid_input.size()[2]
        h = mid_input.size()[3]
        global_input = global_input.unsqueeze(2).unsqueeze(2).expand_as(mid_input)
        fusion_layer = torch.cat((mid_input, global_input), 1)
        fusion_layer = fusion_layer.permute(2, 3, 0, 1).contiguous()
        fusion_layer = fusion_layer.view(-1, 512)
        fusion_layer = self.bn1(self.fc1(fusion_layer))
        fusion_layer = fusion_layer.view(w, h, -1, 256)

        x = fusion_layer.permute(2, 3, 0, 1).contiguous()
        x = F.relu(self.bn2(self.conv1(x)))
        x = self.upsample(x)
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn4(self.conv3(x)))
        x = self.upsample(x)
        #x = torch.nn.functional.tanh(self.bn5(self.conv4(x)))  #final activation function is tanh() for extra credits 
        x = F.sigmoid(self.bn5(self.conv4(x)))
        x = self.upsample(self.conv5(x))
        return x

class colorNet(nn.Module):
    def __init__(self):
        super(colorNet, self).__init__()
        self.low_lv_feat_net = lowLevelFeatNet()
        self.mid_lv_feat_net = midLevelFeatNet()
        self.global_feat_net = globalFeatNet()
        self.class_net = classificationNet()
        self.upsample_col_net = colorizationNet()

    def forward(self, x1, x2):
        x1, x2 = self.low_lv_feat_net(x1, x2)
        print('after low_lv, mid_input is:{}, global_input is:{}'.format(x1.size(), x2.size()))
        x1 = self.mid_lv_feat_net(x1)
        print('after mid_lv, mid2fusion_input is:{}'.format(x1.size()))
        class_input, x2 = self.global_feat_net(x2)
        print('after global_lv, class_input is:{}, global2fusion_input is:{}'.format(class_input.size(), x2.size()))
        class_output = self.class_net(class_input)
        print('after class_lv, class_output is:{}'.format(class_output.size()))
        output = self.upsample_col_net(x1, x2)
        print('after upsample_lv, output is:{}'.format(output.size()))
        return class_output, output
