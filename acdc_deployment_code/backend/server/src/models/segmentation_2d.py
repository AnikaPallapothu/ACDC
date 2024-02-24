from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.InstanceNorm2d(channels)
        #self.bn1 = nn.GroupNorm(channels//4, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.InstanceNorm2d(channels)
       # self.bn2 = nn.GroupNorm(channels//4, channels)
    def forward(self, x):
        initial = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + initial
        return x
class Segnet(nn.Module):
    def __init__(self, num_features=1, channels=[16, 32, 64, 128], num_residual=2, num_outputs=3):
        super(Segnet, self).__init__()
        self.channels = channels
        self._n = len(channels)
        self.convs = [ nn.Conv2d(num_features, channels[0], kernel_size=1) ]
        self.convs += [
            nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1) for i in range(len(channels)-1)
        ]
        self.convs += [
            nn.Conv2d(channels[i], channels[i-1], kernel_size=3, padding=1) for i in range(len(channels)-1, 0, -1)
        ]
        self.convs.append(nn.Conv2d(channels[0], num_outputs, kernel_size=3, padding=1))
        self.residual_layers = [
            nn.Sequential(*[ ResidualBlock(channels[i]) for _ in range(num_residual) ]) for i in range(len(channels))
        ]
        self.residual_layers += [
            nn.Sequential(*[ ResidualBlock(channels[i]) for _ in range(num_residual) ]) for i in range(len(channels)-1, -1, -1)
        ]
        self.pools = [ nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) for _ in range(len(channels)-1) ]
        self.unpools = [ nn.MaxUnpool2d(kernel_size=2, stride=2) for i in range(len(channels)-2, -1, -1)]
        self.seg_outs = [ nn.Conv2d(channels[i], num_outputs, kernel_size=1) for i in range(len(channels)-1, -1, -1)]
        self.convs = nn.ModuleList(self.convs)
        self.residual_layers = nn.ModuleList(self.residual_layers)
        self.seg_outs = nn.ModuleList(self.seg_outs)
        self.unpools = nn.ModuleList(self.unpools)
    def forward(self, x):
        indices = []
        for i in range(self._n-1):
            x = F.relu(self.convs[i](x))
            x, ind = self.pools[i](x)
            indices.append(ind)
            x = self.residual_layers[i](x)        
        x = self.convs[self._n-1](x)
        x = self.residual_layers[self._n-1](x)
        x = self.residual_layers[self._n](x)
        outputs = []
        for i in range(self._n-1):
            j = i + self._n
            k = self._n - i - 2
            if self.training: outputs.append(self.seg_outs[i](x))
            x = F.relu(self.convs[j](x))
            x = self.unpools[i](x, indices[k])
            x = self.residual_layers[j+1](x)        
       # if self.training: self.seg_outs[-1](x)#outputs.append(self.seg_outs[-1](x))
        #else: outputs = self.seg_outs[-1](x)
        outputs = self.seg_outs[-1](x)
        return outputs