import torch
from torch.nn import MSELoss, BCELoss
from model.hash_model.base import BaseCMH
from model.hash_model.backbone import ImgNet, MultiScaleTxtNet, LabNet, Discriminator, LabelDecoder


class SSAH(BaseCMH):
    def __init__(self, dataset, training=True, **kwargs):
        super(SSAH, self).__init__(**kwargs)
        
        self.feat_dim = kwargs['feat_dim']
        self.model_name = '{}_SSAH_{}'.format(dataset, self.bit)
        self.img_net = ImgNet(self.bit)
        self.txt_net = MultiScaleTxtNet(self.txt_dim, self.bit, feat_dim=self.feat_dim)
        self.lab_net = LabNet(self.bit, self.num_class, feat_dim=self.feat_dim)

        self.img_decoder = LabelDecoder(self.num_class, feat_dim=self.feat_dim)
        self.txt_decoder = LabelDecoder(self.num_class, feat_dim=self.feat_dim)
        self.lab_decoder = LabelDecoder(self.num_class, feat_dim=self.feat_dim)

        self.img_dis_net = Discriminator(feat_dim=self.feat_dim)
        self.txt_dis_net = Discriminator(feat_dim=self.feat_dim)

        if training:
            self.alpha = 1
            self.beta = 1
            self.eta = 1
            self.gamma = 1
            self.init_buffer()
            self.mse_loss = MSELoss(reduction='sum')
            self.bce_loss = BCELoss()
    
    def init_buffer(self):
        self.F_buffer = torch.randn(self.num_train, self.bit).cuda()  # store image hash code
        self.G_buffer = torch.randn(self.num_train, self.bit).cuda()  # store text hash code
        self.H_buffer = torch.randn(self.num_train, self.bit).cuda()  # store label hash code

        self.IF_buffer = torch.randn(self.num_train, self.feat_dim).cuda()  # store image feature
        self.TF_buffer = torch.randn(self.num_train, self.feat_dim).cuda()  # store text feature
        self.LF_buffer = torch.randn(self.num_train, self.feat_dim).cuda()  # store label feature

        self.label_buffer = torch.zeros(self.num_train, self.num_class).cuda()

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

    def discrimitator_loss(self, img_feat, txt_feat, lab_feat):
        img_dis_logits = self.img_dis_net(img_feat)
        lab_dis_logits1 = self.img_dis_net(lab_feat)
        txt_dis_logits = self.txt_dis_net(txt_feat)
        lab_dis_logits2 = self.txt_dis_net(lab_feat)
        
        ones_logits = torch.ones_like(img_dis_logits).cuda()
        zeros_logits = torch.zeros_like(img_dis_logits)

        loss_dis = self.bce_loss(img_dis_logits, zeros_logits) \
            + self.bce_loss(lab_dis_logits1, ones_logits) \
            + self.bce_loss(txt_dis_logits, zeros_logits) \
            + self.bce_loss(lab_dis_logits2, ones_logits)

        return loss_dis
    
    @staticmethod
    def calc_neighbor(label1, label2):
        sim = (label1 @ label2.t() > 0).float()
        return sim
    
    def neglog_loss(self, x1, x2, sim):
        theta = 0.5 * (x1 @ x2.t())
        return - torch.sum(sim * theta - torch.log(1.0 + torch.exp(theta)))

    def semantic_loss(self, hash, feat, label_pred, label_gt):
        sim = self.calc_neighbor(label_gt, self.label_buffer)
        feat_logloss = self.neglog_loss(feat, self.LF_buffer, sim)
        hash_logloss = self.neglog_loss(hash, self.H_buffer, sim)
        quan_loss = self.mes_loss(hash, torch.sign(hash).detach())
        rec_loss = self.mse_loss(label_pred, label_gt)

        loss = self.alpha * feat_logloss + self.gamma * hash_logloss + self.eta * quan_loss + self.beta * rec_loss
        return loss
        
    def train_lab_net(self):
        pass

    def train(self, args):
        pass

    def test(self, args):
        pass
