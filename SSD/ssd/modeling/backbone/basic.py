from torch import nn
import torch


class BasicModel(torch.nn.Module):
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

        self.models = nn.ModuleList()
        self.models.append(nn.Sequential(
            self.addConv2D(image_channels, 32),
            torch.nn.BatchNorm2d(32),
            self.addMaxPool2D(),
            nn.ReLU(),
            self.addConv2D(32, 64),
            torch.nn.BatchNorm2d(64),
            self.addMaxPool2D(),
            nn.ReLU(),
            self.addConv2D(64, 64),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(),
            self.addConv2D(64, self.output_channels[0], s=2),
        ))

        self.models.append(nn.Sequential(
            torch.nn.BatchNorm2d(self.output_channels[0]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[0], self.output_channels[0]),
            torch.nn.BatchNorm2d(self.output_channels[0]),
            nn.ReLU(),
            
            #Adding three extra for depth
            self.addConv2D(self.output_channels[0], self.output_channels[0]),
            torch.nn.BatchNorm2d(self.output_channels[0]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[0], self.output_channels[0]),
            torch.nn.BatchNorm2d(self.output_channels[0]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[0], self.output_channels[0]),
            torch.nn.BatchNorm2d(self.output_channels[0]),
            nn.ReLU(),
            
            self.addConv2D(self.output_channels[0], self.output_channels[1], s=2),
        ))

        self.models.append(nn.Sequential(
            torch.nn.BatchNorm2d(self.output_channels[1]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[1], self.output_channels[1]),
            torch.nn.BatchNorm2d(self.output_channels[1]),
            nn.ReLU(),
            
            #Adding three extra for depth
            self.addConv2D(self.output_channels[1], self.output_channels[1]),
            torch.nn.BatchNorm2d(self.output_channels[1]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[1], self.output_channels[1]),
            torch.nn.BatchNorm2d(self.output_channels[1]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[1], self.output_channels[1]),
            torch.nn.BatchNorm2d(self.output_channels[1]),
            nn.ReLU(),
            
            self.addConv2D(self.output_channels[1], self.output_channels[2], s=2),
        ))

        self.models.append(nn.Sequential(
            torch.nn.BatchNorm2d(self.output_channels[2]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[2], self.output_channels[2]),
            torch.nn.BatchNorm2d(self.output_channels[2]),
            nn.ReLU(),
            
            #Adding three extra for depth
            self.addConv2D(self.output_channels[2], self.output_channels[2]),
            torch.nn.BatchNorm2d(self.output_channels[2]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[2], self.output_channels[2]),
            torch.nn.BatchNorm2d(self.output_channels[2]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[2], self.output_channels[2]),
            torch.nn.BatchNorm2d(self.output_channels[2]),
            nn.ReLU(),
            
            self.addConv2D(self.output_channels[2], self.output_channels[3], s=2),
        ))
        
        self.models.append(nn.Sequential(
            torch.nn.BatchNorm2d(self.output_channels[3]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[3], self.output_channels[3]),
            torch.nn.BatchNorm2d(self.output_channels[3]),
            nn.ReLU(),
            
            #Adding three extra for depth
            self.addConv2D(self.output_channels[3], self.output_channels[3]),
            torch.nn.BatchNorm2d(self.output_channels[3]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[3], self.output_channels[3]),
            torch.nn.BatchNorm2d(self.output_channels[3]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[3], self.output_channels[3]),
            torch.nn.BatchNorm2d(self.output_channels[3]),
            nn.ReLU(),
            
            self.addConv2D(self.output_channels[3], self.output_channels[4], s=2),  
        ))
        
        self.models.append(nn.Sequential(
            torch.nn.BatchNorm2d(self.output_channels[4]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[4], self.output_channels[4]),
            torch.nn.BatchNorm2d(self.output_channels[4]),
            nn.ReLU(),
            
            #Adding three extra for depth
            self.addConv2D(self.output_channels[4], self.output_channels[4]),
            torch.nn.BatchNorm2d(self.output_channels[4]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[4], self.output_channels[4]),
            torch.nn.BatchNorm2d(self.output_channels[4]),
            nn.ReLU(),
            self.addConv2D(self.output_channels[4], self.output_channels[4]),
            torch.nn.BatchNorm2d(self.output_channels[4]),
            nn.ReLU(),  
            
            self.addConv2D(self.output_channels[4], self.output_channels[5], padding=0),
        ))
    
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
        last_out_feature = x
        for i in range(8):
            last_out_feature = self.models[i](last_out_feature)
            out_features.append(last_out_feature)

        # out_features.append(self.model0.forward(x)))
        # out_features.append(self.model1.forward(out_features[0])))
        
        #6 lag i cnn
        for idx, feature in enumerate(out_features):
            expected_shape = (self.output_channels[idx], self.output_feature_size[idx], self.output_feature_size[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"

        return tuple(out_features)
    
    def addConv2D(self, image_channels, num_filters, s=1, padding=1):
        return nn.Conv2d(
            in_channels=image_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=s,
            padding=padding
        )

    def addMaxPool2D(self):
        return nn.MaxPool2d(
            kernel_size = 2,
            stride = 2,
        )
    