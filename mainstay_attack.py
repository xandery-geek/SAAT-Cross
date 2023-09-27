import os
import torch
import numpy as np
from tqdm import tqdm
from model.utils import get_attack_model_name, load_model, generate_code, get_database_code
from utils.data import get_data_loader, load_label
from eval.metrics import cal_map, cal_perceptibility
from utils.utils import FileLogger, import_class


def get_generator(name):
    return import_class('model.adv_generator.{}.{}Generator'.format(str.lower(name), name))


def log_results(log_dir, log_file, desc, data):
    logger = FileLogger(log_dir, log_file)
    logger.log(desc)
    for key, val in data:
        logger.log(key + ': {:5f}'.format(val))


def generate_mainstay_code(label, train_code, train_label):
    B = label.size(0)  # batch size
    N = train_label.size(0)  # number of training data

    w_1 = label @ train_label.t()
    label_norm = torch.norm(label, p=2, dim=1, keepdim=True).repeat(1, N)  # B * N
    train_label_norm = torch.norm(train_label, p=2, dim=1, keepdim=True).repeat(1, B) # N * B
    w_1 = w_1 / (label_norm * train_label_norm.t() + 1e-8)  # B * N
    w_2 = 1 - w_1.sign()

    n_p = 1 / torch.sum(w_1, dim=1, keepdim=True)
    w_1 = n_p.where(n_p != torch.inf, torch.tensor([0], dtype=torch.float).cuda()) * w_1

    n_n = 1 / torch.sum(w_2, dim=1, keepdim=True)
    w_2 = n_n.where(n_n != torch.inf, torch.tensor([0], dtype=torch.float).cuda()) * w_2

    code = torch.sign(w_1 @ train_code - w_2 @ train_code)  # B * K
    return code


def select_target_label(data_labels, unique_label):
    """
    select label which is different form original label
    :param data_labels: labels of original datas
    :param unique_label: candidate target labels
    :return: target label for targeted attack
    """
    # remove zero label
    target_label_sum = np.sum(unique_label, axis=1)
    zero_label_idx = np.where(target_label_sum == 0)[0]
    unique_label = np.delete(unique_label, zero_label_idx, axis=0)

    target_idx = []
    similarity = data_labels @ unique_label.transpose()
    for i, _ in enumerate(data_labels):
        s = similarity[i]
        candidate_idx = np.where(s == 0)[0]
        target_idx.append(np.random.choice(candidate_idx, size=1)[0])
    return unique_label[np.array(target_idx)]


def generate_target_label(dataset, test_label, db_label, targeted=False):
    # for non-targeted attack, target_label = test_label
    if not targeted:
        return test_label
    
    # load target label for targeted attack
    target_label_path = 'log/target_label_{}.txt'.format(dataset)
    if os.path.exists(target_label_path):
        target_label = np.loadtxt(target_label_path, dtype=np.int32)
    else:
        print("Generating target labels")
        unique_label = np.unique(db_label, axis=0)
        target_label = select_target_label(test_label, unique_label)
        np.savetxt(target_label_path, target_label, fmt="%d")
    
    return target_label


def batch_preprocess(batch, modality):
    img, txt, _, idx = batch
    if modality == 'image':
        return img.cuda(), idx
    else:
        return txt.cuda(), idx
    
def feature2code(x):
    return x.sign().cpu().detach().numpy()


def attack(model, adv_generator, test_loader, mainstay_codes, modality='image'):
    perceptibility = torch.tensor([0, 0, 0], dtype=torch.float)

    query_code_list, adv_code_list = [], []

    for i, batch in enumerate(tqdm(test_loader)):
        query, idx = batch_preprocess(batch)
        mainstay_code = mainstay_codes[idx]

        batch_size_ = query.size(0)
        adv_query = adv_generator(query, mainstay_code)

        perceptibility += cal_perceptibility(query.cpu().detach(), adv_query.cpu().detach()) * batch_size_

        if modality == 'image':
            query_code = feature2code(model.encode_img(query))
            adv_code = feature2code(model.encode_img(adv_query))
        else:
            query_code = feature2code(model.encode_txt(query))
            adv_code = feature2code(model.encode_txt(adv_query))

        query_code_list.append(query_code)
        adv_code_list.append(adv_code)

    return np.concatenate(query_code_list), \
            np.concatenate(adv_code_list), \
            perceptibility / len(test_loader)


def mainstay_attack(args):
    method = 'Mainstay' + ('_T' if args.targeted else '')
    # load model
    attack_model = get_attack_model_name(args)
    model_path = 'checkpoint/{}.pth'.format(attack_model)
    model = load_model(model_path)

    # load dataset
    database_loader, _ = get_data_loader(args.data_dir, args.dataset, 'database',
                                         args.bs, shuffle=False)
    train_loader, _ = get_data_loader(args.data_dir, args.dataset, 'train',
                                      args.bs, shuffle=True)
    test_loader, _ = get_data_loader(args.data_dir, args.dataset, 'test',
                                     args.bs, shuffle=False)

    # load hashcode and labels
    db_img_codes, db_txt_codes, _ = get_database_code(model, database_loader, attack_model)
    test_label = load_label(args.data_dir, args.dataset, 'test')
    db_label = load_label(args.data_dir, args.dataset, 'database')

    # generate hashcode and labels for training set
    train_img_codes, train_txt_codes, train_labels = generate_code(model, train_loader)
    train_img_codes, train_txt_codes, train_labels = torch.from_numpy(train_img_codes).cuda(), \
        torch.from_numpy(train_txt_codes).cuda(), \
        torch.from_numpy(train_labels).cuda()

    target_label = generate_target_label(args.dataset, test_label, db_label)
    target_label = target_label.cuda()

    # attack image
    adv_generator = get_generator(args.generator)(model, args.img_eps, iteration=args.iteration, targeted=args.targeted)
    mainstay_codes = generate_mainstay_code(target_label, train_txt_codes, train_labels)
    query_code_arr, adv_code_arr, perceptibility = attack(model, adv_generator, test_loader, mainstay_codes, modality='image')

    # save code
    np.save(os.path.join('log', attack_model, 'ori_img_code.npy'), query_code_arr)
    np.save(os.path.join('log', attack_model, '{}_img_code.npy'.format(method)), adv_code_arr)

    # calculate map
    map_dict = {}
    mainstay_code_arr = mainstay_codes.cpu().numpy()

    map_dict['ori_i2t_map'] = cal_map(query_code_arr, test_label, db_txt_codes, db_label)
    map_dict['adv_i2t_map'] = cal_map(adv_code_arr, target_label, db_txt_codes, db_label)
    map_dict['the_i2t_map'] = cal_map(mainstay_code_arr, target_label, db_txt_codes, db_label)
    map_dict['i2t_per'] = perceptibility

    log_results(log_dir=os.path.join('log', attack_model), 
                log_file='{}.txt'.format(method),
                desc='I2T Attack: {}'.format(method),
                data=map_dict)
    
    # attack text
    adv_generator = get_generator(args.generator)(model, args.txt_eps, iteration=args.iteration, targeted=args.targeted)
    mainstay_codes = generate_mainstay_code(target_label, train_img_codes, train_labels)
    query_code_arr, adv_code_arr, perceptibility = attack(model, adv_generator, test_loader, mainstay_codes, modality='text')

    # save code
    np.save(os.path.join('log', attack_model, 'ori_txt_code.npy'), query_code_arr)
    np.save(os.path.join('log', attack_model, '{}_txt_code.npy'.format(method)), adv_code_arr)

    # calculate map
    map_dict = {}
    mainstay_code_arr = mainstay_codes.cpu().numpy()

    map_dict['ori_t2i_map'] = cal_map(query_code_arr, test_label, db_img_codes, db_label)
    map_dict['adv_t2i_map'] = cal_map(adv_code_arr, target_label, db_img_codes, db_label)
    map_dict['the_t2i_map'] = cal_map(mainstay_code_arr, target_label, db_img_codes, db_label)
    map_dict['t2i_per'] = perceptibility

    log_results(log_dir=os.path.join('log', attack_model), 
                log_file='{}.txt'.format(method),
                desc='T2I Attack: {}'.format(method),
                data=map_dict)

