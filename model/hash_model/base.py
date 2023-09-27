import torch.nn as nn
from abc import abstractmethod
from model.hash_model.backbone import AlexNet, VGG, ResNet, TxtModule


class BaseCMH(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.model_name = ''

        # load config
        self.bit = kwargs['bit']
        self.num_train = kwargs['num_train']
        self.num_class = kwargs['num_class']
        self.img_backbone, self.txt_backbone = kwargs['backbone']
        self.txt_dim = kwargs['txt_dim']

    def _build_graph(self):
        if self.img_backbone == 'AlexNet':
            img_net = AlexNet(self.bit)
        elif 'VGG' in self.img_backbone:
            img_net = VGG(self.img_backbone, self.bit)
        else:
            img_net = ResNet(self.img_backbone, self.bit)
        
        txt_net = TxtModule(self.txt_dim, self.bit)

        return img_net, txt_net
    
    @abstractmethod
    def loss_function(self):
        pass
