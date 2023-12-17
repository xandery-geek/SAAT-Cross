import os
import time
import torch
import argparse
import numpy as np
import utils.argument as argument
from tqdm import tqdm
from eval.metrics import cal_map
from model.utils import generate_code
from utils.utils import import_class, setup_seed
from utils.data import get_data_loader, get_train_num, get_classes_num, get_txt_dim

torch.multiprocessing.set_sharing_strategy('file_system')


class Hashing(object):
    def __init__(self, args):

        # load some parameters
        self.args = self.check_args(args)
        self.set_lr()

        # load data and model
        self.train_loader, self.test_loader, self.database_loader = self.load_data()
        self.model = self.load_model()

        # load optimizer
        if args.train:
            self.init_optimizers(self.model)

        # geneare log
        self.model_name = self.model.model_name
        self.log_dir = os.path.join('log', self.model_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.print_log("Model: {}".format(self.model_name))

    def set_lr(self):
        epochs = self.args.epochs
        start, stop = 5e-4, 1e-7
        self.lr_lab = np.linspace(start, stop, epochs + 1)
        self.lr_img = np.linspace(start, stop, epochs + 1)
        self.lr_txt = np.linspace(start, stop, epochs + 1)
        self.lr_dis = np.linspace(start, stop, epochs + 1)

    def check_args(self, args):
        if args.val_interval == -1:
            args.val_interval = args.epochs
        
        return args

    def load_data(self):
        train_loader, _ = get_data_loader(self.args.data_dir, self.args.dataset, 'train', 
                                          batch_size=self.args.bs, shuffle=True)
        test_loader, _ = get_data_loader(self.args.data_dir, self.args.dataset, 'test', 
                                         batch_size=self.args.bs, shuffle=False) 
        database_loader, _ = get_data_loader(self.args.data_dir, self.args.dataset, 'database', 
                                             batch_size=self.args.bs, shuffle=False) 
        
        return train_loader, test_loader, database_loader

    def load_model(self):
        if self.args.load:
            print("Loading Model: {}".format(self.args.ckpt))
            model = torch.load(self.args.ckpt)
        else:
            module = 'model.hash_model.{}.{}'.format(str.lower(self.args.hash_method), self.args.hash_method)
            num_train = get_train_num(self.args.dataset)
            num_class = get_classes_num(self.args.dataset)
            txt_dim = get_txt_dim(self.args.dataset)
            model = import_class(module)(dataset=self.args.dataset,
                                             **{'bit': self.args.bit,
                                                'num_train': num_train,
                                                'num_class': num_class,
                                                'txt_dim': txt_dim,
                                                'feat_dim': 512})
        if torch.cuda.is_available():
            model = model.cuda()

        return model
    
    def init_optimizers(self, model):
        self.lnet_opt = torch.optim.Adam([
            {'params': model.lab_net.parameters()},
            {'params': model.lab_decoder.parameters()}], lr=self.lr_lab[0])
        
        self.inet_opt = torch.optim.Adam([
            {'params': model.img_net.parameters()},
            {'params': model.img_decoder.parameters()}], lr=self.lr_img[0])
        
        self.tnet_opt = torch.optim.Adam([
            {'params': model.txt_net.parameters()},
            {'params': model.txt_decoder.parameters()}], lr=self.lr_txt[0])
        
        self.il_dis_opt = torch.optim.Adam(model.il_dis_net.parameters(), lr=self.lr_dis[0])
        self.tl_dis_opt = torch.optim.Adam(model.tl_dis_net.parameters(), lr=self.lr_dis[0])

    def print_log(self, string, print_time=True):
        if print_time:
            localtime = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
            string = "[" + localtime + '] ' + string
        print(string)
        with open('{}/log.txt'.format(self.log_dir), 'a') as f:
            print(string, file=f)

    def get_dataset(self, split):
        if split == 'train':
            data_loader = self.train_loader
        elif split == 'test':
            data_loader = self.test_loader
        elif split == 'database':
            data_loader = self.database_loader
        else:
            raise ValueError("Unknown dataset {}".format(split))
        return data_loader
    
    @torch.no_grad()
    def generate_code(self, split):
        data_loader = self.get_dataset(split)
        return generate_code(self.model, data_loader)
    
    @staticmethod
    def adjust_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def train_lab_net(self):
        for (_, _, label, idx) in tqdm(self.train_loader):
            label = label.cuda()
            lab_code, lab_feat, lab_pred = self.model.lab_forward(label, idx)
            loss, _ = self.model.semantic_loss(lab_code, lab_feat, lab_pred, label)

            self.lnet_opt.zero_grad()
            loss.backward()
            self.lnet_opt.step()

        total_loss = self.model.total_loss(modality='lab')
        return total_loss.item()
    
    def train_img_net(self):
        tmp_list = []
        for (img, _, label, idx) in tqdm(self.train_loader):
            img, label = img.cuda(), label.cuda()
            img_code, img_feat, img_pred = self.model.img_forward(img, idx, label)

            loss, loss_items = self.model.semantic_loss(img_code, img_feat, img_pred, label) \
                #  + self.model.il_dis_loss(img_feat, is_label=True)
            
            self.inet_opt.zero_grad()
            loss.backward()
            self.inet_opt.step()
            
            tmp_list.append(np.array(loss_items))
        
        tmp_arr = np.array(tmp_list)
        print(tmp_arr.mean(axis=0))

        total_loss = self.model.total_loss(modality='img')
        return total_loss.item()
    
    def train_txt_net(self):
        tmp_list = []
        for (_, txt, label, idx) in tqdm(self.train_loader):
            txt, label = txt.cuda(), label.cuda()
            txt_code, txt_feat, txt_pred = self.model.txt_forward(txt, idx)

            loss, loss_items = self.model.semantic_loss(txt_code, txt_feat, txt_pred, label) \
                #  + self.model.tl_dis_loss(txt_feat, is_label=True)
            
            self.tnet_opt.zero_grad()
            loss.backward()
            self.tnet_opt.step()

            tmp_list.append(np.array(loss_items))
        
        tmp_arr = np.array(tmp_list)
        print(tmp_arr.mean(axis=0))

        total_loss = self.model.total_loss(modality='txt')
        return total_loss.item()
    
    def train_dis_net(self):
        total_loss = 0
        for (img, txt, label, idx) in tqdm(self.train_loader):
            img, txt, label = img.cuda(), txt.cuda(), label.cuda()

            _, img_feat = self.model.img_net(img)
            _, txt_feat = self.model.txt_net(txt)
            _, lab_feat = self.model.lab_net(label)

            loss = self.model.il_dis_loss(img_feat.detach(), is_label=False) \
                + self.model.il_dis_loss(lab_feat.detach(), is_label=True) \
                + self.model.tl_dis_loss(txt_feat.detach(), is_label=False) \
                + self.model.tl_dis_loss(lab_feat.detach(), is_label=True)
        
            self.il_dis_opt.zero_grad()
            self.tl_dis_opt.zero_grad()
            loss.backward()
            self.il_dis_opt.step()
            self.tl_dis_opt.step()

            total_loss += loss.item()
        return total_loss / (len(self.train_loader))

    def training(self):
        self.print_log('>>>Training Model<<<')
        self.model.train()
        
        loss_recorder = {
            'labnet': [],
            'imgnet': [],
            'txtnet': [],
            'disnet': [],
        }

        for epoch in range(self.args.epochs):
            self.print_log('==>Training Epoch: {}'.format(epoch))

            # for _ in range(self.args.K_lab):
            #     loss = self.train_lab_net()
            #     self.print_log('LabNet loss: {:.5f}'.format(loss))
            # loss_recorder['labnet'].append(loss)
            
            # for _ in range(self.args.K_dis):
            #     loss = self.train_dis_net()
            #     self.print_log('DisNet loss: {:.5f}'.format(loss))
            # loss_recorder['disnet'].append(loss)

            for _ in range(self.args.K_img):
                loss = self.train_img_net()
                self.print_log('ImgNet loss: {:.5f}'.format(loss))
            loss_recorder['imgnet'].append(loss)

            # for _ in range(self.args.K_txt):
            #     loss = self.train_txt_net()
            #     self.print_log('TxtNet loss: {:.5f}'.format(loss))
            # loss_recorder['txtnet'].append(loss)

            self.model.update_hash_codes()

            self.adjust_learning_rate(self.inet_opt, self.lr_img[epoch+1])
            # self.adjust_learning_rate(self.lnet_opt, self.lr_lab[epoch+1])
            # self.adjust_learning_rate(self.il_dis_opt, self.lr_dis[epoch+1])
            # self.adjust_learning_rate(self.tl_dis_opt, self.lr_dis[epoch+1])
            # self.adjust_learning_rate(self.tnet_opt, self.lr_txt[epoch+1])

            if (epoch + 1) % self.args.val_interval == 0:
                self.test()
                self.model.train()

        torch.save(self.model, os.path.join(self.args.ckpt, self.model_name + '.pth'))

        from plot.plot import plot_loss_curve
        plot_loss_curve(loss_recorder, self.log_dir + '/loss.jpg')

    def test(self):
        self.print_log('>>>Testing MAP<<<')
        test_img_codes, test_txt_codes, test_labels = self.generate_code('test')
        db_img_codes, db_txt_codes, db_labels = self.generate_code('database')

        # img2txt = cal_map(test_img_codes, test_labels, db_txt_codes, db_labels)
        # txt2img = cal_map(test_txt_codes, test_labels, db_img_codes, db_labels)
        # self.print_log("Img2Txt MAP: {:.5f}".format(img2txt))
        # self.print_log("Txt2Img MAP: {:.5f}".format(txt2img))

        img2img = cal_map(test_img_codes, test_labels, db_img_codes, db_labels)
        # txt2txt = cal_map(test_txt_codes, test_labels, db_txt_codes, db_labels)
        self.print_log("Img2Img MAP: {:.5f}".format(img2img))
        # self.print_log("Txt2Txt MAP: {:.5f}".format(txt2txt))

    def generate(self):
        self.print_log('>>>Generating hash code for database<<<')
        if not args.load:
            self.model = self.load_model()
        db_img_codes, db_txt_codes, db_labels = self.generate_code('database')
        db_img_codes, db_txt_codes, db_labels = db_img_codes.astype(int), db_txt_codes.astype(int), db_labels.astype(int)

        print("Writing hash code of database to {}".format(self.log_dir))
        np.save(os.path.join(self.log_dir, 'db_img_codes.npy'), db_img_codes)
        np.save(os.path.join(self.log_dir, 'db_txt_codes.npy'), db_txt_codes)
        np.save(os.path.join(self.log_dir, 'db_labels.npy'), db_labels)

        with open(os.path.join(self.log_dir, 'database.txt'), 'w') as f:
            for i in range(len(db_labels)):
                print('{},{},{},{}'.format(i, ' '.join(map(str, db_img_codes[i])), 
                                           ' '.join(map(str, db_txt_codes[i])),
                                           ' '.join(map(str, db_labels[i]))), file=f)


def parser_arguments():
    parser = argparse.ArgumentParser()
    
    parser = argument.add_base_arguments(parser)
    parser = argument.add_dataset_arguments(parser)

    parser.add_argument('--hash_method', dest='hash_method', default='SSAH', 
                        help='names of deep hashing methods')
    parser.add_argument('--bit', dest='bit', type=int, default=32, help='length of the hashing code')

    # arguments for different phases
    parser.add_argument('--train', dest='train', action='store_true', default=False, help='to train or not')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='to test or not')

    # arguments for training
    parser.add_argument('--load', dest='load', action='store_true', default=False, help='load the latest model for continue training')
    parser.add_argument('--ckpt', dest='ckpt', default='checkpoint', help='models are saved here')
    parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='number of epoch')
    parser.add_argument('--K_lab', dest='K_lab', type=int, default=1, help='training iterations in each epoch')
    parser.add_argument('--K_img', dest='K_img', type=int, default=3, help='training iterations in each epoch')
    parser.add_argument('--K_txt', dest='K_txt', type=int, default=5, help='training iterations in each epoch')
    parser.add_argument('--K_dis', dest='K_dis', type=int, default=1, help='training iterations in each epoch')
    parser.add_argument('--val_interval', dest='val_interval', type=int, default=10, help='interval for validation')

    parser.add_argument('--bs', dest='bs', type=int, default=128, help='number of images in one batch')
    
    return parser.parse_args()


if __name__ == '__main__':
    setup_seed(seed=1)

    args = parser_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    hashing = Hashing(args)
    if args.train:
        hashing.training()
    if args.test:
        hashing.test()
