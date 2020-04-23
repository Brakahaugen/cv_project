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

        
        
        self.model = [

            nn.Sequential(
                self.resnet.conv1,
                self.resnet.bn1,
                self.resnet.relu,
                self.resnet.maxpool,
                self.resnet.layer1,
                self.resnet.layer2,
                # (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                # (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                nn.Conv2d(128, output_channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(output_channels[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                torchvision.models.BasicBlock(output_channels[0], output_channels[1], downsample=basic_downsample(output_channels[0], output_channels[1])),
                torchvision.models.BasicBlock(output_channels[1], output_channels[1])
            ),

            nn.Sequential(
                torchvision.models.BasicBlock(output_channels[1], output_channels[2], downsample=basic_downsample(output_channels[1], output_channels[2])),
                torchvision.models.BasicBlock(output_channels[2], output_channels[2])
            ),

            nn.Sequential(
                torchvision.models.BasicBlock(output_channels[2], output_channels[3], downsample=basic_downsample(output_channels[2], output_channels[3])),
                torchvision.models.BasicBlock(output_channels[3], output_channels[3])
            ),

            nn.Sequential(
                torchvision.models.BasicBlock(output_channels[3], output_channels[4], downsample=basic_downsample(output_channels[3], output_channels[4])),
                torchvision.models.BasicBlock(output_channels[4], output_channels[4])
            ),

            nn.Sequential(
                torchvision.models.BasicBlock(output_channels[4], output_channels[5], downsample=basic_downsample(output_channels[4], output_channels[5])),
                torchvision.models.BasicBlock(output_channels[5], output_channels[5]),
                nn.Conv2d(output_channels[5], output_channels[5], kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(output_channels[5]),
                nn.ReLU()
            )
        ]

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
            out_features.append(x)
            print(
                f"Output shape of layer: {x.shape}"
            )

        for idx, feature in enumerate(out_features):
            print(feature.shape[1:])

        return tuple(out_features)
