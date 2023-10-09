import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


vgg_dict = {"VGG11": models.vgg11, "VGG13": models.vgg13, "VGG16": models.vgg16, "VGG19": models.vgg19,
            "VGG11BN": models.vgg11_bn, "VGG13BN": models.vgg13_bn, "VGG16BN": models.vgg16_bn,
            "VGG19BN": models.vgg19_bn}

resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class AlexNet(nn.Module):
    def __init__(self, bit):
        super(AlexNet, self).__init__()
        original_model = models.alexnet(pretrained=True)
        self.features = original_model.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = original_model.classifier[1].weight
        cl1.bias = original_model.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = original_model.classifier[4].weight
        cl2.bias = original_model.classifier[4].bias

        self.classifier = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, bit),
        )

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1.0):
        x = (x - self.mean) / self.std
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        y = self.tanh(alpha * y)
        return y


class VGG(nn.Module):
    def __init__(self, model_name, bit):
        super(VGG, self).__init__()
        vgg_model = vgg_dict[model_name](pretrained=True)
        self.features = vgg_model.features
        self.cl1 = nn.Linear(25088, 4096)
        self.cl1.weight = vgg_model.classifier[0].weight
        self.cl1.bias = vgg_model.classifier[0].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = vgg_model.classifier[3].weight
        cl2.bias = vgg_model.classifier[3].bias

        self.classifier = nn.Sequential(
            self.cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, bit),
        )

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1.0):
        x = (x - self.mean) / self.std
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        y = self.tanh(alpha * y)
        return y


class ResNet(nn.Module):
    def __init__(self, model_name, hash_bit):
        super(ResNet, self).__init__()
        resnet_model = resnet_dict[model_name](pretrained=True)
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2,
                                            self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(resnet_model.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

        self.activation = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1.0):
        x = (x - self.mean) / self.std
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        y = self.activation(alpha * y)
        return y


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)


class TxtModule(nn.Module):
    def __init__(self, input_dim, hash_bit, mid_dim=8192):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TxtModule, self).__init__()
        self.module_name = "text_model"

        # full-conv layers
        self.conv1 = nn.Conv2d(1, mid_dim, kernel_size=(input_dim, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(mid_dim, hash_bit, kernel_size=1, stride=(1, 1))
    
        self.activation = nn.Tanh()
        self.apply(weights_init)

    def forward(self, x, alpha=1.0):
        x = x.unsqueeze(1).unsqueeze(-1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        y = x.squeeze()
        y = self.activation(alpha * y)
        return y


class ImgNet(nn.Module):
    def __init__(self, hash_bit, model_name='ResNet18'):
        super(ImgNet, self).__init__()
        resnet_model = resnet_dict[model_name](pretrained=True)
        feature_layers = [resnet_model.conv1, 
                          resnet_model.bn1, 
                          resnet_model.relu, 
                          resnet_model.maxpool,
                          resnet_model.layer1,
                          resnet_model.layer2,
                          resnet_model.layer3,
                          resnet_model.layer4,
                          resnet_model.avgpool]
        self.feature_layers = nn.Sequential(*feature_layers)

        self.fc = nn.Conv2d(in_channels=resnet_model.fc.in_features, out_channels=hash_bit, kernel_size=1)

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x):
        x = (x - self.mean) / self.std
        feature = self.feature_layers(x)
        hash = self.tanh(self.fc(feature))
        return hash.squeeze(), feature.squeeze()


def init_parameters_recursively(layer):
    if isinstance(layer, nn.Sequential):
        for sub_layer in layer:
            init_parameters_recursively(sub_layer)
    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, std=0.01)
        if layer.bias is not None:
            nn.init.normal_(layer.bias, std=0.01)
    else:
        return


class ScaleBlock(nn.Module):
    def __init__(self, scale, n_filters=100):
        super(ScaleBlock, self).__init__()
        kernel_size = (1, 5 * scale)
        stride = (1, 5 * scale)
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=kernel_size, stride=stride, padding=0),
            nn.ReLU()
        )
    
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.layers)

    def forward(self, x):
        x = self.layers(x) # (B, 1, 1, L) -> (B, n_filters, 1, L/kernel)
        x = F.avg_pool2d(x, (1, x.shape[3])) # (B, n_filters, 1, 1)
        return x


class MultiScaleTxt(nn.Module):
    def __init__(self, n_filters, scales):
        super(MultiScaleTxt, self).__init__()

        self.module_list = nn.ModuleList([
            ScaleBlock(scale, n_filters) for scale in scales])

    def forward(self, x):
        y = []
        for module in self.module_list:
            y.append(module(x))
        return torch.cat(y, dim=1)


class MultiScaleTxtNet(nn.Module):
    def __init__(self, bit, feat_dim=512):
        super(MultiScaleTxtNet, self).__init__()
        scales = [1, 2, 3, 6, 10]
        n_filters = 100
        self.ms_module = MultiScaleTxt(n_filters, scales)
        self.features = nn.Sequential(
            nn.Conv2d(len(scales) * n_filters, 4096, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(4096, feat_dim, kernel_size=1),
            nn.ReLU()
        )

        self.hash_layers = nn.Sequential(
            nn.Conv2d(feat_dim, bit, kernel_size=1),
            nn.Tanh(),
        )

        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.features)
        init_parameters_recursively(self.hash_layers)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(1)
        x = self.ms_module(x)
        feature = self.features(x)
        hash = self.hash_layers(feature)
        
        return hash.squeeze(), feature.squeeze()


class LabNet(nn.Module):
    def __init__(self, bit, num_class, feat_dim=512):
        super(LabNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4096, kernel_size=(1, num_class)),
            nn.ReLU(),
            nn.modules.normalization.LocalResponseNorm(size=4, alpha=0.0001, beta=0.75, k=2.0),
            nn.Conv2d(4096, feat_dim, kernel_size=1),
            nn.ReLU()
        )
        self.hash_layers = nn.Sequential(
            nn.Conv2d(feat_dim, bit, kernel_size=1),
            nn.Tanh(),
        )

        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.features)
        init_parameters_recursively(self.hash_layers)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(1)
        feature = self.features(x)
        hash = self.hash_layers(feature)
        
        return hash.squeeze(), feature.squeeze()


class LabelDecoder(nn.Module):
    def __init__(self, num_class, feat_dim=512):
        super(LabelDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(feat_dim, num_class, kernel_size=1),
            nn.Sigmoid()
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.layers)
    
    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.layers(x)
        return x.squeeze()


class Discriminator(nn.Module):
    def __init__(self, feat_dim=512) -> None:
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim, feat_dim//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim//2, 1, kernel_size=1)
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.layers)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.layers(x)

        return x.squeeze()
