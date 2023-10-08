import torch
from mainstay_attack import generate_mainstay_code, get_generator
from model.utils import load_model, generate_code_ordered, get_victim_model_name
from utils.utils import check_dir
from utils.data import get_data_loader, get_classes_num


def adv_loss(adv_code, target_code):
    loss = torch.mean(adv_code * target_code)
    return loss


def mainstay_training(args):
    print("=> lambda: {}, mu: {}".format(args.lam, args.mu))


    victim_model_name = get_victim_model_name(args)
    model = load_model('checkpoint/{}.pth'.format(victim_model_name))


    train_loader, num_train = get_data_loader(args.data_dir, args.dataset, 'train',
                                      batch_size=args.bs, shuffle=True)
    
    print("Generating codes and labels of training set")
    train_img_codes, train_txt_codes, train_labels = generate_code_ordered(model, train_loader, num_train, args.bit,
                                                    get_classes_num(args.dataset))
    train_img_codes, train_txt_codes, train_labels = torch.from_numpy(train_img_codes).cuda(), \
        torch.from_numpy(train_txt_codes).cuda(), \
        torch.from_numpy(train_labels).cuda()

    model.train()
    # initialize buffers of hashing model
    if hasattr(model, 'F_buffer') and hasattr(model, 'G_buffer') and hasattr(model, 'label_buffer'):
        model.F_buffer.data = train_img_codes.data
        model.G_buffer.data = train_txt_codes.data
        model.label_buffer.data = train_labels.data
        model.update_hash_codes()

    optimizer_img = torch.optim.SGD(model.img_net.parameters(), lr=0.03, weight_decay=1e-5)
    optimizer_txt = torch.optim.SGD(model.txt_net.parameters(), lr=0.03, weight_decay=1e-5)
    lr_steps = args.epochs * len(train_loader)
    scheduler_img = torch.optim.lr_scheduler.MultiStepLR(optimizer_img, milestones=[lr_steps/2, lr_steps*3/4], gamma=0.1)
    scheduler_txt = torch.optim.lr_scheduler.MultiStepLR(optimizer_txt, milestones=[lr_steps/2, lr_steps*3/4], gamma=0.1)

    img_generator = get_generator(args.generator)(model, args.img_eps, iteration=args.iteration)
    txt_generator = get_generator(args.generator)(model, args.txt_eps, iteration=args.iteration)

    # adversarial training
    for epoch in range(args.epochs):
        img_loss = 0
        for i, (img, txt, label, idx) in enumerate(train_loader):
            img, txt, label = img.cuda(), txt.cuda(), label.cuda()

            # inner minimization aims to generate adversarial examples
            mainstay_code = generate_mainstay_code(label, model.G_buffer, train_labels)
            adv_img = img_generator(img, mainstay_code, 'image')

            # outer maximization aims to optimize parameters of model
            model.zero_grad()
            adv_code = model.encode_img(adv_img)
            loss_ben = model.loss_function((img, txt, label, idx), 0)
            loss_adv = - adv_loss(adv_code, mainstay_code)
            loss_qua = torch.mean((adv_code - torch.sign(adv_code)) ** 2)
            loss = args.lam * loss_adv + args.mu * loss_qua + loss_ben
            loss.backward()
            optimizer_img.step()
            scheduler_img.step()
            img_loss += loss.item()

            if i % 50 == 0:
                print("loss: {:.5f}\tben: {:.5f}\tadv: {:.5f}\tqua: {:.5f}".format(loss, loss_ben.item(),
                                                                     loss_adv.item(), loss_qua.item()))

        print('Epoch: %3d/%3d\tImage training loss: %3.5f \n' % (epoch, args.epochs, img_loss/len(train_loader)))

        txt_loss = 0
        for i, (img, txt, label, idx) in enumerate(train_loader):
            img, txt, label = img.cuda(), txt.cuda(), label.cuda()

            # inner minimization aims to generate adversarial examples
            mainstay_code = generate_mainstay_code(label, model.F_buffer, train_labels)
            adv_txt = txt_generator(txt, mainstay_code, 'text')

            # outer maximization aims to optimize parameters of model
            model.zero_grad()
            adv_code = model.encode_txt(adv_txt)
            loss_ben = model.loss_function((img, txt, label, idx), 1)
            loss_adv = - adv_loss(adv_code, mainstay_code)
            loss_qua = torch.mean((adv_code - torch.sign(adv_code)) ** 2)
            loss = args.lam * loss_adv + args.mu * loss_qua + loss_ben
            loss.backward()
            optimizer_txt.step()
            scheduler_txt.step()
            txt_loss += loss.item()

            if i % 50 == 0:
                print("loss: {:.5f}\tben: {:.5f}\tadv: {:.5f}\tqua: {:.5f}".format(loss, loss_ben.item(),
                                                                     loss_adv.item(), loss_qua.item()))
        model.update_hash_codes()

        print('Epoch: %3d/%3d\tText training loss: %3.5f \n' % (epoch, args.epochs, txt_loss/len(train_loader)))

    robust_model_name = 'mainstay_{}'.format(victim_model_name)

    check_dir('log/{}'.format(robust_model_name))
    robust_model_path = 'checkpoint/{}.pth'.format(robust_model_name)
    torch.save(model, robust_model_path)
