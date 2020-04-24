import torchvision
from torch import nn

def basic_downsample(inplanes, planes): 
    return nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

class ResnetModel(nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        
        self.resnet = torchvision.models.resnet34(pretrained=cfg.MODEL.BACKBONE.PRETRAINED)

        
        
        self.model = nn.ModuleList([
            nn.Sequential(
                self.resnet.conv1,
                self.resnet.bn1,
                self.resnet.relu,
                self.resnet.layer1,
                self.resnet.layer2,
                #out: 128
            ),
            
            nn.Sequential(
                self.resnet.layer3,
                #in: 128 out:256
#                 torchvision.models.resnet.BasicBlock(output_channels[2], output_channels[2])
            ),

            nn.Sequential(
                self.resnet.layer4,
                #in: 256 out: 512
#                 torchvision.models.resnet.BasicBlock(output_channels[2], output_channels[2])
            ),
            #     torchvision.models.resnet.BasicBlock(output_channels[0], output_channels[1], stride = (2, 2), downsample=basic_downsample(output_channels[0], output_channels[1])),
            #     torchvision.models.resnet.BasicBlock(output_channels[1], output_channels[1])
            # ),

            nn.Sequential(
                torchvision.models.resnet.BasicBlock(output_channels[2], output_channels[3], stride = (2, 2), downsample=basic_downsample(output_channels[2], output_channels[3])),
                torchvision.models.resnet.BasicBlock(output_channels[3], output_channels[3])
            ),

            nn.Sequential(
                torchvision.models.resnet.BasicBlock(output_channels[3], output_channels[4], stride = (2, 2), downsample=basic_downsample(output_channels[3], output_channels[4])),
                torchvision.models.resnet.BasicBlock(output_channels[4], output_channels[4])
            ),

            nn.Sequential(
                torchvision.models.resnet.BasicBlock(output_channels[4], output_channels[5], stride = (2, 2), downsample=basic_downsample(output_channels[4], output_channels[5])),
                torchvision.models.resnet.BasicBlock(output_channels[5], output_channels[5])
            ),

            nn.Sequential(
#                 # 2x3
                torchvision.models.resnet.BasicBlock(output_channels[5], output_channels[6], stride = (2, 2), downsample=basic_downsample(output_channels[5], output_channels[6])),
                torchvision.models.resnet.BasicBlock(output_channels[6], output_channels[6]),

#                 # 1x2
                nn.Conv2d(output_channels[5], output_channels[6], kernel_size=(3, 4), padding=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(output_channels[6]),
                nn.ReLU(inplace=True)
            )
        ])
#         torch.Size([256, 30, 40])
# torch.Size([512, 15, 20])
# torch.Size([256, 8, 10])
# torch.Size([256, 4, 5])
# torch.Size([128, 2, 3])
# torch.Size([64, 2, 2])

        # self.model = nn.ModuleList([self.model])
        print(self.model)

    def forward(self, x):
        """
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        for layer in self.model:
            x = layer(x)
            out_features.append(x)
#             print(
#                 f"Output shape of layer: {x.shape}"
#             )

#         for idx, feature in enumerate(out_features):
#             print(feature.shape[1:])

        return tuple(out_features)
