import torchvision
import torch


class ResnetModel(torch.nn.Module):
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

        
        self.model = torchvision.models.resnet34(pretrained=cfg.MODEL.BACKBONE.PRETRAINED)

        print(self.model)
        for layer in self.model:
            if type(layer) == torch.nn.Conv2d:
                print(layer)

                layer.shape
        self.model.conv1.shape[0] = cfg.defaults.output_channels[0]


        self.sequential_layers = [
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,  # Ouput 256 x 40 x 30
            self.model.layer4,
        ]


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []


        for layer in self.sequential_layers:
            x = layer(x)
            out_features.append(x)
            print(
                f"Output shape of layer: {x.shape}"
            )

        for idx, feature in enumerate(out_features):
            print(feature.shape[1:])

        return tuple(out_features)

        # output = self.model(x)
        # print(output)
        # print(output.shape)
        # return output

