import os
import torch
import numpy as np
import argparse
import utils.argument as argument
from tqdm import tqdm
from model.utils import get_victim_model_name, load_model, generate_code, get_database_code
from utils.data import get_data_loader, load_label
from utils.utils import FileLogger, import_class
from eval.metrics import cal_hamming_dis, cal_map, cal_perceptibility


def get_generator(name):
    return import_class('model.adv_generator.{}.{}Generator'.format(str.lower(name), name))


def log_results(log_dir, log_file, desc, data):
    logger = FileLogger(log_dir, log_file)
    logger.log(desc)
    for key, val in data.items():
        if isinstance(val, float):
            logger.log(key + ': {:5f}'.format(val))
        else:
            logger.log(key + ': {}'.format(val))


def generate_mainstay_code(label, train_code, train_label):
    label, train_label = label.float(), train_label.float()
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
    num_test = 0

    for i, batch in enumerate(tqdm(test_loader)):
        query, idx = batch_preprocess(batch, modality)
        mainstay_code = mainstay_codes[idx]

        batch_size = query.size(0)
        num_test += batch_size
        adv_query = adv_generator(query, mainstay_code, modality)

        perceptibility += cal_perceptibility(query.cpu().detach(), adv_query.cpu().detach()) * batch_size

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
            perceptibility / num_test


def load_dataset(args):
    db_loader, _ = get_data_loader(args.data_dir, args.dataset, 'database',
                                         batch_size=args.bs, shuffle=False)
    train_loader, _ = get_data_loader(args.data_dir, args.dataset, 'train',
                                      batch_size=args.bs, shuffle=True)
    test_loader, _ = get_data_loader(args.data_dir, args.dataset, 'test',
                                     batch_size=args.bs, shuffle=False)
    
    return train_loader, test_loader, db_loader


def mainstay_attack(args):
    attack_name = 'Mainstay' + ('_T' if args.targeted else '')
    # load model
    victim_model = get_victim_model_name(args)
    model = load_model('checkpoint/{}.pth'.format(victim_model))

    # load dataset
    train_loader, test_loader, db_loader = load_dataset(args)

    # load hashcode and labels
    db_img_codes, db_txt_codes, _ = get_database_code(model, db_loader, victim_model)
    test_label = load_label(args.data_dir, args.dataset, 'test')
    db_label = load_label(args.data_dir, args.dataset, 'database')

    # generate hashcode and labels for training set
    train_img_codes, train_txt_codes, train_labels = generate_code(model, train_loader)
    train_img_codes, train_txt_codes, train_labels = torch.from_numpy(train_img_codes).cuda(), \
        torch.from_numpy(train_txt_codes).cuda(), \
        torch.from_numpy(train_labels).cuda()

    target_label = generate_target_label(args.dataset, test_label, db_label)

    # attack image
    adv_generator = get_generator(args.generator)(model, args.img_eps, iteration=args.iteration, targeted=args.targeted)
    mainstay_codes = generate_mainstay_code(torch.from_numpy(target_label).cuda(), train_txt_codes, train_labels)
    query_code_arr, adv_code_arr, perceptibility = attack(model, adv_generator, test_loader, mainstay_codes, modality='image')

    # save code
    np.save(os.path.join('log', victim_model, 'ori_img_code.npy'), query_code_arr)
    np.save(os.path.join('log', victim_model, '{}_img_code.npy'.format(attack_name)), adv_code_arr)

    # calculate map
    map_dict = {}
    mainstay_code_arr = mainstay_codes.cpu().numpy() if args.targeted else - mainstay_codes.cpu().numpy()

    map_dict['ori_i2t_map'] = cal_map(query_code_arr, test_label, db_txt_codes, db_label, top_k=None)
    map_dict['adv_i2t_map'] = cal_map(adv_code_arr, target_label, db_txt_codes, db_label, top_k=None)
    map_dict['the_i2t_map'] = cal_map(mainstay_code_arr, target_label, db_txt_codes, db_label, top_k=None)
    map_dict['i2t_per'] = perceptibility

    log_results(log_dir=os.path.join('log', victim_model), 
                log_file='{}.txt'.format(attack_name),
                desc='I2T Attack: {}'.format(attack_name),
                data=map_dict)
    
    # attack text
    adv_generator = get_generator(args.generator)(model, args.txt_eps, iteration=args.iteration, targeted=args.targeted)
    mainstay_codes = generate_mainstay_code(torch.from_numpy(target_label).cuda(), train_img_codes, train_labels)
    query_code_arr, adv_code_arr, perceptibility = attack(model, adv_generator, test_loader, mainstay_codes, modality='text')

    # save code
    np.save(os.path.join('log', victim_model, 'ori_txt_code.npy'), query_code_arr)
    np.save(os.path.join('log', victim_model, '{}_txt_code.npy'.format(attack_name)), adv_code_arr)

    # calculate map
    map_dict = {}
    mainstay_code_arr = mainstay_codes.cpu().numpy() if args.targeted else - mainstay_codes.cpu().numpy()

    map_dict['ori_t2i_map'] = cal_map(query_code_arr, test_label, db_img_codes, db_label, top_k=None)
    map_dict['adv_t2i_map'] = cal_map(adv_code_arr, target_label, db_img_codes, db_label, top_k=None)
    map_dict['the_t2i_map'] = cal_map(mainstay_code_arr, target_label, db_img_codes, db_label, top_k=None)
    map_dict['t2i_per'] = perceptibility

    log_results(log_dir=os.path.join('log', victim_model), 
                log_file='{}.txt'.format(attack_name),
                desc='T2I Attack: {}'.format(attack_name),
                data=map_dict)


def ann_retrieve(query_codes, db_codes, top):
    # calculate top index
    retrieval_indices = []
    for query in query_codes:
        hamming_dis = cal_hamming_dis(query, db_codes)
        sort_idx = np.argsort(hamming_dis)
        retrieval_indices.append(sort_idx[:top])

    return retrieval_indices


def save_retrieval_results(filename, query_id, retrieval_id):

    data = []
    data.append('query id:[retrieval id]')
    for i in len(query_id):
        data.append('{}:{}'.format(query_id[i], ','.join(map(str, retrieval_id[i]))))

    with open(filename, 'w') as f:
        f.writelines(data)


def sample_or_retrieve(args):
    victim_model = get_victim_model_name(args)
    model = load_model('checkpoint/{}.pth'.format(victim_model))

    # load dataset
    train_loader, test_loader, db_loader = load_dataset(args)
    
    db_img_codes, db_txt_codes, _ = get_database_code(model, db_loader, victim_model)
    test_label = load_label(args.data_dir, args.dataset, 'test')
    db_label = load_label(args.data_dir, args.dataset, 'database')

    # generate hashcode and labels for training set
    train_img_codes, train_txt_codes, train_labels = generate_code(model, train_loader)
    train_img_codes, train_txt_codes, train_labels = torch.from_numpy(train_img_codes).cuda(), \
        torch.from_numpy(train_txt_codes).cuda(), \
        torch.from_numpy(train_labels).cuda()

    target_label = generate_target_label(args.dataset, test_label, db_label)
    
    # attack image
    adv_generator = get_generator(args.generator)(model, args.img_eps, iteration=args.iteration, targeted=args.targeted)
    mainstay_codes = generate_mainstay_code(torch.from_numpy(target_label).cuda(), train_txt_codes, train_labels)
    
    modality = args.modality
    save_pth = os.path.join('log', victim_model)
    for i, batch in enumerate(tqdm(test_loader)):
        if i == args.batch_id:
            query, idx = batch_preprocess(batch, modality)
            query, idx = query[:args.sample_num], idx[:args.sample_num]
            mainstay_code = mainstay_codes[idx]

            adv_query = adv_generator(query, mainstay_code, modality)

            if args.sample:
                print("Sampling data of {}-th batch".format(i))

                ori_data, adv_data = query.cpu().numpy(), adv_query.cpu().numpy()
                np.save(save_pth + '/ori-{}-{}-{}.npy'.format(modality, idx[0], idx[-1]), ori_data)
                np.save(save_pth + '/adv-{}-{}-{}.npy'.format(modality, idx[0], idx[-1]), adv_data)

            if args.retrieve:
                print("Retrieving data of {}-th batch".format(i))

                if modality == 'image':
                    query_code = feature2code(model.encode_img(query))
                    adv_code = feature2code(model.encode_img(adv_query))
                    ori_results = ann_retrieve(query_code, db_txt_codes)
                    adv_results = ann_retrieve(adv_code, db_txt_codes)
                else:
                    query_code = feature2code(model.encode_txt(query))
                    adv_code = feature2code(model.encode_txt(adv_query))
                    ori_results = ann_retrieve(query_code, db_img_codes)
                    adv_results = ann_retrieve(adv_code, db_img_codes)

                save_retrieval_results(save_pth + '/ori-{}-{}-{}.txt'.format(modality, idx[0], idx[-1]), idx, ori_results)
                save_retrieval_results(save_pth + '/adv-{}-{}-{}.txt'.format(modality, idx[0], idx[-1]), idx, adv_results)

            break

def parser_arguments():
    parser = argparse.ArgumentParser()
    
    parser = argument.add_base_arguments(parser)
    parser = argument.add_dataset_arguments(parser)
    parser = argument.add_model_arguments(parser)
    parser = argument.add_attack_arguments(parser)
    
    # arguments for defense
    parser.add_argument('--adv', dest='adv', action="store_true", default=False,
                        help='load model with adversarial training')
    parser = argument.add_defense_arguments(parser)

    # arguments for dataset
    parser.add_argument('--bs', dest='bs', type=int, default=128, help='number of images in one batch')
    
    # arguments for sampling or retrieval
    parser.add_argument('--batch_id', dest='batch_id', type=int, default=0, help='batch index for sampling or retrieval')
    parser.add_argument('--modality', dest='modality', default='image', choices=['image', 'text'], help='query modality')
    parser.add_argument('--sample_num', dest='sample_num', type=int, default=10, help='number of sampling')
    parser.add_argument('--retrieval_num', dest='retrieval_num', type=int, default=10, help='number of retrieval')

    return parser.parse_args()


if __name__ == '__main__':

    args = parser_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    print("Current Method: {}".format(args.attack_method))
    if args.sample or args.retrieve:
        sample_or_retrieve(args)
    else:
        mainstay_attack(args)
