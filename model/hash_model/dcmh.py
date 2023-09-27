import torch
import numpy as np
from model.hash_model.base import BaseCMH


class DCMH(BaseCMH):
    def __init__(self, dataset, training=True, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model_name = '{}_DCMH_{}-{}_{}'.format(dataset, self.img_backbone, self.txt_backbone, self.bit)
        self.img_net, self.txt_net = self._build_graph()

        self.eta = 0.1
        self.gamma = 0.1

        if training:
            self.init_buffer()

    def init_buffer(self):
        self.F_buffer = torch.randn(self.num_train, self.bit)  # store image feature
        self.G_buffer = torch.randn(self.num_train, self.bit)  # store text feature
        self.label_buffer = torch.zeros(self.num_train, self.num_class)

        self.F_buffer = self.F_buffer.cuda()
        self.G_buffer = self.G_buffer.cuda()
        self.label_buffer = self.label_buffer.cuda()

        self.B = torch.sign(self.F_buffer + self.G_buffer)
    
    def update_hash_codes(self):
        self.B = torch.sign(self.F_buffer + self.G_buffer)

    def encode_img(self, img, alpha=1):
        return self.img_net(img, alpha)
    
    def encode_txt(self, txt, alpha=1):
        return self.txt_net(txt, alpha)

    def forward(self, img, txt, alpha=1):
        img_code = self.img_net(img, alpha)
        txt_code = self.txt_net(txt, alpha)
        return img_code, txt_code

    @staticmethod
    def calc_neighbor(label1, label2):
        sim = (label1 @ label2.t() > 0).float()
        return sim
    
    def loss_function(self, batch, optimizer_idx):
        if optimizer_idx == 0:
            # train image net
            image, _, label, idx = batch
            unupdated_idx = np.setdiff1d(range(self.num_train), idx)
            batch_size = image.size(0)
            
            cur_f = self.encode_img(image)  # cur_f: (batch_size, bit)
            self.F_buffer[idx, :] = cur_f.data
            self.label_buffer[idx, :] = label
            F, G = self.F_buffer, self.G_buffer

            S = self.calc_neighbor(label, self.label_buffer)  # S: (batch_size, num_train)
            theta_x = 0.5 * (cur_f @ G.t())
            logloss_x = - torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))
            quantization_x = torch.sum(torch.pow(self.B[idx, :] - cur_f, 2))
            balance_x = torch.sum(torch.pow(torch.sum(cur_f, dim=0) + torch.sum(F[unupdated_idx], dim=0), 2))
            loss_x = logloss_x + self.gamma * quantization_x + self.eta * balance_x
            loss_x /= (batch_size * self.num_train)

            return loss_x
        else:
            # train text net
            _, text, label, idx = batch
            unupdated_idx = np.setdiff1d(range(self.num_train), idx)
            batch_size = text.size(0)
            
            cur_g = self.encode_txt(text)  # cur_f: (batch_size, bit)
            self.G_buffer[idx, :] = cur_g.data
            self.label_buffer[idx, :] = label
            F, G = self.F_buffer, self.G_buffer

            S = self.calc_neighbor(label, self.label_buffer)  # S: (batch_size, num_train)
            theta_y = 0.5 * (cur_g @ F.t())
            logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))
            quantization_y = torch.sum(torch.pow(self.B[idx, :] - cur_g, 2))
            balance_y = torch.sum(torch.pow(torch.sum(cur_g, dim=0) + torch.sum(G[unupdated_idx], dim=0), 2))
            loss_y = logloss_y + self.gamma * quantization_y + self.eta * balance_y
            loss_y /= (batch_size * self.num_train)

            return loss_y

    def total_loss(self):
        B, F, G = self.B, self.F_buffer, self.G_buffer
        sim = self.calc_neighbor(self.label_buffer, self.label_buffer)

        theta = 0.5 * (F @ G.t())
        term1 = torch.sum(torch.log(1 + torch.exp(theta)) - sim * theta)
        term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
        term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
        loss = term1 + self.gamma * term2 + self.eta * term3
        loss /= (self.num_train * self.num_train)
        return loss
