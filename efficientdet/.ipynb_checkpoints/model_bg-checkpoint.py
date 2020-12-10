import torch
import torch.nn as nn
import numpy as np

class PredBG(nn.Module):
    def __init__(self, in_channel, feat, num_bg, device):
        super(PredBG, self).__init__()
        activation=nn.LeakyReLU(0.3)
        num_pred_layer_for_bg = 4
        base_feature_for_pred = 4
        pad = int((4 - 1) / 2)
        bg_init = np.zeros([1, num_bg, in_channel, feat, feat])
        bg_tensor = torch.tensor(bg_init, dtype=torch.float32, requires_grad=True, device=device)
        self.bg_tensor = bg_tensor
        self.num_bg = num_bg
        self.bg_conv = nn.Sequential()
        for i in range(num_pred_layer_for_bg):
            out_channel = base_feature_for_pred * 2 ** i
            self.bg_conv.add_module("background_conv_%d" % (i+1), 
                                    nn.Conv2d(in_channel, out_channel, 4, 2, pad))
            self.bg_conv.add_module("background_bn_%d" % (i+1),
                                    nn.BatchNorm2d(out_channel))
            self.bg_conv.add_module("background_rl_%d" % (i+1),
                                    activation)
            in_channel = out_channel
        
        fc_in_channel = int((feat / 2**num_pred_layer_for_bg) ** 2 * in_channel)
        self.fc_in_channel = fc_in_channel
        self.bg_fc = nn.Sequential()
        self.bg_fc.add_module("background_fc_1", nn.Linear(fc_in_channel, 512))
        self.bg_fc.add_module("background_fc_2", nn.Linear(512, 128))
        self.bg_fc.add_module("background_fc_3", nn.Linear(128, num_bg))
        self.bg_fc.add_module("background_softmax", nn.Softmax(dim=1))
        
    def forward(self, feature):
        feature_conv = self.bg_conv(feature)
        feature_fc_input = feature_conv.view(-1, self.fc_in_channel)
        bg_ratio = self.bg_fc(feature_fc_input) #[batch_size, num_bg]
        bg_ratio = bg_ratio.view(-1, self.num_bg, 1, 1, 1)        
        bg_aggregate = bg_ratio * self.bg_tensor  
        bg_final = bg_aggregate.sum(dim=1)
        bg_final = bg_final.sum(dim=0, keepdims=True) / feature.size(0)
        bg_final = bg_final.clamp(0.0, 1.0) 
        return bg_ratio, bg_final
        
        