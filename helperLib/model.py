from torch import nn, Tensor
import torch
from torchvision import models


class vanilla_autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Encoder from vgg16
        vgg16_model = models.vgg16(weights="IMAGENET1K_V1")
        self.encoder_vgg16 = nn.Sequential(*list(vgg16_model.features.children())[:-1])
        for param in self.encoder_vgg16.parameters():
            param.requires_grad = False

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=(3, 3), padding="same"
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=(3, 3), padding="same"
            ),
            nn.ReLU(),
            nn.UpsamplingNearest2d(size=(28, 28)),
            nn.Conv2d(
                in_channels=128, out_channels=64, kernel_size=(3, 3), padding="same"
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=32, kernel_size=(3, 3), padding="same"
            ),
            nn.ReLU(),
            nn.UpsamplingNearest2d(size=(56, 56)),
            nn.Conv2d(
                in_channels=32, out_channels=16, kernel_size=(3, 3), padding="same"
            ),
            nn.ReLU(),
            nn.UpsamplingNearest2d(size=(112, 112)),
            nn.Conv2d(
                in_channels=16, out_channels=2, kernel_size=(3, 3), padding="same"
            ),
            nn.Tanh(),
            nn.UpsamplingNearest2d(size=(224, 224)),
        )

    def forward(self, x):
        x = self.encoder_vgg16(x)
        x = self.decoder(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down
        for feature in features:
            self.down.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        # Up
        for feature in reversed(features):
            self.up.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                ),
            )
            self.up.append(DoubleConv(in_channels=feature * 2, out_channels=feature))

        # Bottleneck
        self.bottleneck = DoubleConv(
            in_channels=features[-1], out_channels=features[-1] * 2
        )

        # Final 1x1 Conv
        self.final_conv = nn.Conv2d(
            in_channels=features[0], out_channels=out_channels, kernel_size=1
        )

    def forward(self, x: Tensor):
        skip_connections = []

        # Down
        for down in self.down:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Up
        skip_connections = skip_connections[::-1]
        for index in range(0, len(self.up), 2):
            # ConvTranspose2d
            x = self.up[index](x)

            # Concat Skip connection with x
            skip_connection = skip_connections[index // 2]
            concat_x = torch.cat((skip_connection, x), dim=1)

            # DoubleConv
            x = self.up[index + 1](concat_x)

        return self.final_conv(x)


def getTrainedModelDict() -> dict[str, nn.Module]:
    modelDict: dict[str, nn.Module] = {}

    # autoencoder first draft
    newModel: nn.Module = vanilla_autoencoder()
    checkpoint: dict = torch.load(
        "./checkpoint/autoencoder_mse_50epoch0.pth", map_location=torch.device("cpu")
    )
    newModel.load_state_dict(checkpoint["model_state_dict"])
    modelDict["auto0"] = newModel

    # autoencoder full
    newModel: nn.Module = vanilla_autoencoder()
    checkpoint: dict = torch.load(
        "./checkpoint/autoencoder_mse_50epoch.pth", map_location=torch.device("cpu")
    )
    newModel.load_state_dict(checkpoint["model_state_dict"])
    modelDict["auto"] = newModel

    # unet 5 epoch
    newModel = UNET()
    checkpoint = torch.load(
        "./checkpoint/unet_mse_5epoch.pth", map_location=torch.device("cpu")
    )
    newModel.load_state_dict(checkpoint["model_state_dict"])
    modelDict["unet5"] = newModel

    # unet 10 epoch
    newModel = UNET()
    checkpoint = torch.load(
        "./checkpoint/unet_mse_10epoch.pth", map_location=torch.device("cpu")
    )
    newModel.load_state_dict(checkpoint["model_state_dict"])
    modelDict["unet10"] = newModel

    # unet 50 epoch
    newModel = UNET()
    checkpoint = torch.load(
        "./checkpoint/unet_mse_50epoch.pth", map_location=torch.device("cpu")
    )
    newModel.load_state_dict(checkpoint["model_state_dict"])
    modelDict["unet50"] = newModel

    # unet 60 epoch
    newModel = UNET()
    checkpoint = torch.load(
        "./checkpoint/unet_mse_60epoch.pth", map_location=torch.device("cpu")
    )
    newModel.load_state_dict(checkpoint["model_state_dict"])
    modelDict["unet60"] = newModel

    return modelDict


if __name__ == "__main__":
    readyModel: dict[str, nn.Module] = getTrainedModelDict()
