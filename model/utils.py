import os
import time
import torch
import numpy as np
from tqdm import tqdm


def load_model(path):
    print("Loading {}".format(path))
    model = torch.load(path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def get_victim_model_name(args):
    attack_model = '{}_{}_{}-{}_{}'.format(args.dataset, args.hash_method, args.img_backbone, args.txt_backbone, args.bit)
    if args.adv:
        attack_model = '{}_{}'.format(args.adv_method, attack_model)
        if args.p_lambda != 1.0 or args.p_mu != 1e-4:
            attack_model = '{}_{}_{}'.format(attack_model, args.p_lambda, args.p_mu)
    return attack_model


def generate_code(model, data_loader):
    img_code_list, txt_code_list, labels_list = [], [], []

    model.eval()
    for img, txt, label, _ in tqdm(data_loader):
        img, txt = img.cuda(), txt.cuda()
        img_code, txt_code = model(img, txt)
        img_code_list.append(img_code.data.cpu())
        txt_code_list.append(txt_code.data.cpu())
        labels_list.append(label)
    return torch.cat(img_code_list).sign().numpy(), \
            torch.cat(txt_code_list).sign().numpy(), \
            torch.cat(labels_list).numpy()


def generate_code_ordered(model, data_loader, num_data, bit, num_class):
    img_codes = torch.zeros([num_data, bit])
    txt_codes = torch.zeros([num_data, bit])
    labels = torch.zeros(num_data, num_class)

    for img, txt, label, idx in data_loader:
        img, txt = img.cuda(), txt.cuda()
        img_code, txt_code = model(img, txt)
        img_codes[idx, :] = img_code.data.cpu()
        txt_codes[idx, :] = txt_code.data.cpu()
        labels[idx, :] = label
    return img_codes.sign().numpy(), txt_codes.sign().numpy(), labels.numpy()


def get_database_code(model, dataloader, attack_model):
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    database_path = 'log/{}'.format(attack_model)
    db_img_codes_file = os.path.join(database_path, 'db_img_codes.npy')
    db_txt_codes_file = os.path.join(database_path, 'db_txt_codes.npy')
    db_labels_file = os.path.join(database_path, 'db_labels.npy')

    if os.path.exists(db_img_codes_file) \
        and os.path.exists(db_txt_codes_file) \
        and os.path.exists(db_labels_file):
        
        # check time stamp
        code_stamp1 = get_time_stamp(db_img_codes_file)
        code_stamp2 = get_time_stamp(db_txt_codes_file)
        label_stamp = get_time_stamp(db_labels_file)
        model_stamp = get_time_stamp(model_path)

        if model_stamp < code_stamp1 and model_stamp < code_stamp2 and model_stamp < label_stamp:
            print("Loading\n"
                  "image codes: {}\n"
                  "text codes: {}\n"
                  "labels: {}".format(db_img_codes_file, db_txt_codes_file, db_labels_file))
            
            db_img_codes = np.load(db_img_codes_file)
            db_txt_codes = np.load(db_txt_codes_file)
            db_labels = np.load(db_labels_file)
            return db_img_codes, db_txt_codes, db_labels

    print("Generating database code")
    db_img_codes, db_txt_codes, db_labels = generate_code(model, dataloader)
    if not os.path.exists(database_path):
        os.makedirs(database_path)
    np.save(db_img_codes_file, db_img_codes)
    np.save(db_txt_codes_file, db_txt_codes)
    np.save(db_labels_file, db_labels)
    return db_img_codes, db_txt_codes, db_labels


def get_time_stamp(file):
    stamp = os.stat(file).st_mtime
    return time.localtime(stamp)


def get_alpha(cur_epoch, epochs):
    alpha = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    epoch2alpha = epochs / len(alpha)
    return alpha[int(cur_epoch / epoch2alpha)]
