import torch
import torch.nn as nn
from model.hash_model.backbone import ImgNet, MultiScaleTxtNet, LabNet, Discriminator, LabelDecoder


class SSAH(nn.Module):
    def __init__(self, dataset, training=True, **kwargs):
        super(SSAH, self).__init__()
        
        self.bit = kwargs['bit']
        self.num_train = kwargs['num_train']
        self.num_class = kwargs['num_class']
        self.txt_dim = kwargs['txt_dim']
        self.feat_dim = kwargs['feat_dim']
        self.model_name = '{}_SSAH_{}'.format(dataset, self.bit)

        self.img_net = ImgNet(self.bit)
        self.txt_net = MultiScaleTxtNet(self.bit, feat_dim=self.feat_dim)
        self.lab_net = LabNet(self.bit, self.num_class, feat_dim=self.feat_dim)

        self.img_decoder = LabelDecoder(self.num_class, feat_dim=self.feat_dim)
        self.txt_decoder = LabelDecoder(self.num_class, feat_dim=self.feat_dim)
        self.lab_decoder = LabelDecoder(self.num_class, feat_dim=self.feat_dim)

        self.il_dis_net = Discriminator(feat_dim=self.feat_dim)
        self.tl_dis_net = Discriminator(feat_dim=self.feat_dim)

        if training:
            self.alpha = 1
            self.beta = 1
            self.eta = 1
            self.gamma = 1
            self.init_buffer()
            self.mse_loss = nn.MSELoss(reduction='mean')
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    
    def init_buffer(self):
        self.F_buffer = torch.randn(self.num_train, self.bit).cuda()  # store image hash code
        self.G_buffer = torch.randn(self.num_train, self.bit).cuda()  # store text hash code
        self.H_buffer = torch.randn(self.num_train, self.bit).cuda()  # store label hash code

        self.IF_buffer = torch.randn(self.num_train, self.feat_dim).cuda()  # store image feature
        self.TF_buffer = torch.randn(self.num_train, self.feat_dim).cuda()  # store text feature
        self.LF_buffer = torch.randn(self.num_train, self.feat_dim).cuda()  # store label feature

        self.label_buffer = torch.zeros(self.num_train, self.num_class).cuda()

        self.B = torch.sign(self.F_buffer + self.G_buffer + self.H_buffer)

    def update_hash_codes(self):
        self.B = torch.sign(self.F_buffer + self.G_buffer + self.H_buffer)
    
    def encode_img(self, img):
        img_code, _ = self.img_net(img)
        return img_code
    
    def encode_txt(self, txt):
        txt_code, _ = self.txt_net(txt)
        return txt_code

    def forward(self, img, txt):
        img_code, _ = self.img_net(img)
        txt_code, _ = self.txt_net(txt)
        return img_code, txt_code
    
    def img_forward(self, img, idx, lab):
        code, feat = self.img_net(img)
        label = self.img_decoder(feat)

        self.F_buffer[idx, :] = code.data
        self.IF_buffer[idx, :] = feat.data
        self.label_buffer[idx, :] = lab.data
        return code, feat, label
    
    def txt_forward(self, txt, idx):
        code, feat = self.txt_net(txt)
        label = self.txt_decoder(feat)

        self.G_buffer[idx, :] = code.data
        self.TF_buffer[idx, :] = feat.data
        return code, feat, label
    
    def lab_forward(self, lab, idx):
        code, feat = self.lab_net(lab)
        label = self.lab_decoder(feat)

        self.H_buffer[idx, :] = code.data
        self.LF_buffer[idx, :] = feat.data
        self.label_buffer[idx, :] = lab.data
        return code, feat, label
    
    def tl_dis_loss(self, feat, is_label=True):
        logits = self.tl_dis_net(feat)
        
        if is_label:
            logits_truth = torch.ones_like(logits).cuda()
        else:
            logits_truth = torch.zeros_like(logits).cuda()

        loss_dis = self.bce_loss(logits, logits_truth) 

        return loss_dis
    
    def il_dis_loss(self, feat, is_label=True):
        logits = self.il_dis_net(feat)
        
        if is_label:
            logits_truth = torch.ones_like(logits).cuda()
        else:
            logits_truth = torch.zeros_like(logits).cuda()

        loss_dis = self.bce_loss(logits, logits_truth) 

        return loss_dis
    
    @staticmethod
    def calc_neighbor(label1, label2):
        sim = (label1 @ label2.t() > 0).float()
        return sim
    
    def neglog_loss(self, x1, x2, sim):
        theta = 0.5 * (x1 @ x2.t())
        return self.mse_loss(sim * theta, nn.functional.softplus(theta))

    def semantic_loss(self, hash, feat, label_pred, label_gt):
        sim = self.calc_neighbor(label_gt, self.label_buffer)
        feat_logloss = self.neglog_loss(feat, self.IF_buffer, sim)
        hash_logloss = self.neglog_loss(hash, self.F_buffer, sim)
        quan_loss = self.mse_loss(hash, torch.sign(hash).detach())
        rec_loss = self.mse_loss(label_pred, label_gt)

        loss = self.alpha * feat_logloss + self.gamma * hash_logloss + self.eta * quan_loss + self.beta * rec_loss
        return loss, [feat_logloss.item(), hash_logloss.item(), quan_loss.item(), rec_loss.item()]
    
    @torch.no_grad()
    def total_loss(self, modality='img'):
        if modality == 'img':
            feat = self.IF_buffer
            hash = self.F_buffer
        elif modality == 'txt':
            feat = self.TF_buffer
            hash = self.G_buffer
        else:
            feat = self.LF_buffer
            hash = self.H_buffer

        sim = self.calc_neighbor(self.label_buffer, self.label_buffer)
        feat_logloss = self.neglog_loss(feat, self.IF_buffer, sim)
        hash_logloss = self.neglog_loss(hash, self.F_buffer, sim)
        quan_loss = self.mse_loss(hash, torch.sign(hash).detach())

        loss = self.alpha * feat_logloss + self.gamma * hash_logloss + self.eta * quan_loss
        return loss