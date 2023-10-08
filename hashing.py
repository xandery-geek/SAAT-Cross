import os
import time
import torch
import argparse
import numpy as np
import utils.argument as argument
from tqdm import tqdm
from utils.data import get_data_loader, get_train_num, get_classes_num, get_txt_dim
from utils.utils import import_class, setup_seed
from eval.metrics import cal_map
from model.utils import generate_code


torch.multiprocessing.set_sharing_strategy('file_system')


class Hashing(object):

    training_strategy ={
        'epoch': None,
        'batch': None
    }

    def __init__(self, args):
        self.register()

        # load some parameters
        self.args = self.check_args(args)
        self.lr = args.lr

        # load data and model
        self.train_loader, self.test_loader, self.database_loader = self.load_data()
        self.model = self.load_model()

        # load optimizer
        if args.train:
            self.optimizer_img, self.scheduler_img = self.load_optimizers(
                self.model.img_net.parameters(), self.lr)
            self.optimizer_txt, self.scheduler_txt = self.load_optimizers(
                self.model.txt_net.parameters(), self.lr)

        # geneare log
        self.model_name = self.model.model_name
        self.log_dir = os.path.join('log', self.model_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.print_log("Model: {}".format(self.model_name))
    
    def register(self):
        self.training_strategy['epoch'] = self.train_by_epoch
        self.training_strategy['batch'] = self.train_by_batch

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
                                                'backbone': (self.args.img_backbone, self.args.txt_backbone),
                                                'num_train': num_train,
                                                'num_class': num_class,
                                                'txt_dim': txt_dim})
        if torch.cuda.is_available():
            model = model.cuda()

        return model
    
    def load_optimizers(self, params, lr, **kwargs):
        optimizer = torch.optim.SGD(params, lr=lr, **kwargs)
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args.epochs)
        return optimizer, scheduler

    def print_log(self, string, print_time=True):
        if print_time:
            localtime = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
            string = "[" + localtime + '] ' + string
        print(string)
        with open('{}/log.txt'.format(self.log_dir), 'a') as f:
            print(string, file=f)

    def train_by_batch(self, epoch):
        self.print_log('Train Epoch {}: [lr]={:.5f}'.format(epoch, self.lr))

        process = tqdm(self.train_loader)
        for i, (img, txt, label, idx) in enumerate(process):
            img, txt, label = img.cuda(), txt.cuda(), label.cuda()

            # update image netas
            loss1 = self.model.loss_function((img, txt, label, idx), 0)
            self.optimizer_img.zero_grad()
            loss1.backward()
            self.optimizer_img.step()

            loss2 = self.model.loss_function((img, txt, label, idx), 1)
            self.optimizer_txt.zero_grad()
            loss2.backward()
            self.optimizer_txt.step()

            self.model.update_hash_codes()

        with torch.no_grad():
            avg_loss = self.model.total_loss().item()

        self.print_log("loss: {:.5f}".format(avg_loss))
        return avg_loss
    
    def train_by_epoch(self, epoch):
        self.print_log('Train Epoch {}: [lr]={:.5f}'.format(epoch, self.lr))

        process = tqdm(self.train_loader)
        for i, (img, txt, label, idx) in enumerate(process):
            img, txt, label = img.cuda(), txt.cuda(), label.cuda()

            # update image netas
            loss1 = self.model.loss_function((img, txt, label, idx), 0)
            self.optimizer_img.zero_grad()
            loss1.backward()
            self.optimizer_img.step()

        for i, (img, txt, label, idx) in enumerate(process):
            img, txt, label = img.cuda(), txt.cuda(), label.cuda()

            loss2 = self.model.loss_function((img, txt, label, idx), 1)
            self.optimizer_txt.zero_grad()
            loss2.backward()
            self.optimizer_txt.step()

        self.model.update_hash_codes()

        with torch.no_grad():
            avg_loss = self.model.total_loss().item()

        self.print_log("loss: {:.5f}".format(avg_loss))
        return avg_loss

    def adjust_learning_rate(self):
        self.scheduler_img.step()
        self.scheduler_txt.step()
        self.lr = self.scheduler_img.get_last_lr()[0]

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

    def load_database(self):
        hash_code_arr = np.load(os.path.join(self.log_dir, 'database_hashcode.npy'))
        labels_arr = np.load(os.path.join(self.log_dir, 'database_label.npy'))
        return hash_code_arr, labels_arr

    def training(self):
        self.print_log('>>>Training Model<<<')

        train_func = self.training_strategy[self.args.strategy]
        record_loss = []
        self.model.train()

        for epoch in range(0, self.args.epochs):
            loss = train_func(epoch)
            record_loss.append(loss)
            self.adjust_learning_rate()

            # if (epoch + 1) % self.args.val_interval == 0:        
        torch.save(self.model, os.path.join(self.args.ckpt, self.model_name + '.pth'))
        self.test()

    def test(self):
        self.print_log('>>>Testing MAP<<<')
        test_img_codes, test_txt_codes, test_labels = self.generate_code('test')
        db_img_codes, db_txt_codes, db_labels = self.generate_code('database')

        img2txt = cal_map(test_img_codes, test_labels, db_txt_codes, db_labels)
        txt2img = cal_map(test_txt_codes, test_labels, db_img_codes, db_labels)
        self.print_log("Img2Txt MAP: {:.5f}".format(img2txt))
        self.print_log("Txt2Img MAP: {:.5f}".format(txt2img))

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

    # def retrieve(self, batch=0, top=10):
    #     self.print_log('>>>Retrieve relevant images<<<')
    #     if not args.load:
    #         self.model = self.load_model()
    #     self.model.eval()

    #     # get data batch
    #     images, labels, _ = get_batch(self.test_loader, batch)
    #     # calculate hash code
    #     outputs = self.model(images.cuda())
    #     outputs = outputs.data.cpu()
    #     database_codes, _ = self.load_database()

    #     images_arr, labels_arr = retrieve_images(images.numpy(), labels.numpy(), outputs, database_codes, top,
    #                                              args.data_dir, args.dataset)

    #     print("Writing retrieve images of database to {}".format(self.log_dir))
    #     np.save(os.path.join(self.log_dir, 'retrieve_images_{}.npy'.format(batch)), images_arr)
    #     np.save(os.path.join(self.log_dir, 'retrieve_labels_{}.npy'.format(batch)), labels_arr)
    

def parser_arguments():
    parser = argparse.ArgumentParser()
    
    parser = argument.add_base_arguments(parser)
    parser = argument.add_dataset_arguments(parser)
    parser = argument.add_model_arguments(parser)

    # arguments for different phases
    parser.add_argument('--train', dest='train', action='store_true', default=False, help='to train or not')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='to test or not')
    parser.add_argument('--generate', dest='generate', action='store_true', default=False, help='to generate or not')
    parser.add_argument('--retrieve', dest='retrieve', action='store_true', default=False, help='to retrieve or not')

    # arguments for training
    parser.add_argument('--load', dest='load', action='store_true', default=False, help='load the latest model for continue training')
    parser.add_argument('--ckpt', dest='ckpt', default='checkpoint', help='models are saved here')
    parser.add_argument('--epochs', dest='epochs', type=int, default=500, help='number of epoch')
    parser.add_argument('--lr', dest='lr', type=float, default=0.03, help='initial learning rate for SGD')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--wd', dest='wd', type=float, default=5e-4, help='weight decay for SGD')
    parser.add_argument('--val_interval', dest='val_interval', type=int, default=-1, help='interval for validation')
    parser.add_argument('--strategy', dest='strategy', type=str, default='batch', choices=['batch', 'epoch'], help='training strategy')

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
    if args.generate:
        hashing.generate()
    if args.retrieve:
        hashing.retrieve()
