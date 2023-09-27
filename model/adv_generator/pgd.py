import torch
from model.utils import get_alpha
from model.adv_generator.base import BaseGenerator


class PGDGenerator(BaseGenerator):
    def __init__(self, model, eps, step=1.0, iteration=100, targeted=False, record_loss=False):
        super().__init__('PGD', model, eps, targeted, record_loss)

        self.step = step
        self.iteration = iteration

    def image_forward(self, images, target_code):
        delta = torch.zeros_like(images).cuda()
        delta.uniform_(-self.eps, self.eps)
        delta.data = (images.data + delta.data).clamp(0, 1) - images.data
        delta.requires_grad = True

        loss_list = [] if self.record_loss else None
        for i in range(self.iteration):
            alpha = get_alpha(i, self.iteration)
            adv_code = self.model.encode_img(images + delta, alpha)
            loss = self.loss(adv_code, target_code.detach())
            loss.backward()

            delta.data = delta - self.step / 255 * torch.sign(delta.grad.detach())
            delta.data = delta.data.clamp(-self.eps, self.eps)
            delta.data = (images.data + delta.data).clamp(0, 1) - images.data
            delta.grad.zero_()

            if loss_list is not None and (i + 1) % (self.iteration // 10) == 0:
                loss_list.append(round(loss.item(), 4))

        if loss_list is not None:
            print("loss :{}".format(loss_list))
        return images + delta.detach()

    def text_forward(self, text, target_code):
        delta = torch.zeros_like(text).cuda()
        delta.uniform_(-self.eps, self.eps)
        delta.data = (text.data + delta.data).clamp(0, 1) - text.data
        delta.requires_grad = True

        loss_list = [] if self.record_loss else None
        for i in range(self.iteration):
            alpha = get_alpha(i, self.iteration)
            adv_code = self.model.encode_txt(text + delta, alpha)
            loss = self.loss(adv_code, target_code.detach())
            loss.backward()

            delta.data = delta - self.step / 255 * torch.sign(delta.grad.detach())
            delta.data = delta.data.clamp(-self.eps, self.eps)
            delta.data = (text.data + delta.data).clamp(0, 1) - text.data
            delta.grad.zero_()

            if loss_list is not None and (i + 1) % (self.iteration // 10) == 0:
                loss_list.append(round(loss.item(), 4))

        if loss_list is not None:
            print("loss :{}".format(loss_list))
        return text + delta.detach()

    def forward(self, images, target_code, modality='image'):
        if modality == 'image':
            return self.image_forward(images, target_code)
        else:
            return self.text_forward(images, target_code)