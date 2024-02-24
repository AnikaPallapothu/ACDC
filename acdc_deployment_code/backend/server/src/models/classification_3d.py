from torch import nn
import torch 

class Bottleneck3D(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels, stride=(1,1,1), downsample=False):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, intermediate_channels, kernel_size=(1,1,1), padding=0, bias=False)
        self.bn1 = nn.InstanceNorm3d(intermediate_channels)
        self.conv2 = nn.Conv3d(intermediate_channels, intermediate_channels, kernel_size=(3,3,3), stride=stride, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(intermediate_channels)
        self.conv3 = nn.Conv3d(intermediate_channels, out_channels, kernel_size=(1,1,1), padding=0, bias=False)
        self.bn3 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), padding=0, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels)
            )
        else:
            self.downsample = None
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x + residual
        return x
        

class Resnet50_3d(nn.Module):

    def __init__(self):
        super(Resnet50_3d, self).__init__()   
        self.conv1 = nn.Conv3d(4,64, kernel_size=(3,3,3), stride=1, padding=(1,1,1), bias=False)
        self.bn1 = nn.InstanceNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool2 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2))
        
        self.layer1 = self._make_layer(in_channels=64, intermediate_channels=64, out_channels=256, num_blocks=3, stride=(1,1,1), downsample=True)
        self.layer2 = self._make_layer(in_channels=256, intermediate_channels=128, out_channels=512, num_blocks=4, stride=(2,2,2),downsample=True)
        self.layer3 = self._make_layer(in_channels=512, intermediate_channels=256, out_channels=1024, num_blocks=6, stride=(2,2,2),downsample=True)
        self.layer4 = self._make_layer(in_channels=1024, intermediate_channels=512, out_channels=2048, num_blocks=3, stride=(2,2,2),downsample=True)
        
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(2048, 10)   # classes = 6 because each lobe has value/score ranging [0-5]
        self.fc2 = nn.Linear(20, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 10)
        self.fc5 = nn.Linear(20, 5)

    def _make_layer(self, in_channels, intermediate_channels, out_channels, num_blocks, stride, downsample=False):
        layers = [
                Bottleneck3D(in_channels=in_channels, intermediate_channels=intermediate_channels, out_channels=out_channels, stride=stride, downsample=downsample),
            ]
        layers += [ Bottleneck3D(in_channels=out_channels, intermediate_channels=intermediate_channels, out_channels=out_channels) for _ in range(1, num_blocks) ]
        return nn.Sequential(*layers)

    def forward(self, x, y):
        features = []      
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
        x = x.reshape(x.shape[0], -1)
        x1 = self.fc1(x)
        x2 = self.fc2(y)
        x2 = self.fc3(x2)
        x2 = self.fc4(x2)
        x = torch.cat([x1, x2], 1)
        x = self.fc5(x)

        return x