from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

class L1_plus_perceptualLoss(nn.Module):
    def __init__(self, lambda_L1, lambda_perceptual, perceptual_layers, gpu_ids, percep_is_l1):
        super(L1_plus_perceptualLoss, self).__init__()

        self.lambda_L1 = lambda_L1
        self.lambda_perceptual = lambda_perceptual
        self.gpu_ids = gpu_ids

        self.percep_is_l1 = percep_is_l1

        # vgg = models.vgg19(pretrained=True).features
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load('/home/usuaris/imatge/jorge.pueyo/RoomGAN/vgg/vgg19-dcbb9e9d.pth'))
        vgg = vgg19.features


        self.vgg_submodel = nn.Sequential()
        for i,layer in enumerate(list(vgg)):
            self.vgg_submodel.add_module(str(i),layer)
            if i == perceptual_layers:
                break
        self.vgg_submodel = torch.nn.DataParallel(self.vgg_submodel, device_ids=gpu_ids).cuda()


        #TODO: Hardcoded perceptual layers for second perceptual loss
        PERCEPTUAL_LAYERS_2 = 14
        self.vgg_submodel_2 = nn.Sequential()
        for i, layer in enumerate(list(vgg)):
            self.vgg_submodel_2.add_module(str(i),layer)
            if i == PERCEPTUAL_LAYERS_2:
                break
        self.vgg_submodel_2 = torch.nn.DataParallel(self.vgg_submodel_2, device_ids=gpu_ids).cuda()


        print(self.vgg_submodel)

    def forward(self, inputs, targets_1, targets_2):
        if self.lambda_L1 == 0 and self.lambda_perceptual == 0:
            return Variable(torch.zeros(1)).cuda(), Variable(torch.zeros(1)), Variable(torch.zeros(1))
        # normal L1
        #loss_l1 = F.l1_loss(inputs, targets) * self.lambda_L1
        loss_l1 = F.l1_loss(inputs, targets_2) * self.lambda_L1

        #TODO: Mean and std are hardcoded
        # perceptual L1
        """
        mean = torch.FloatTensor(3)
        mean[0] = 0.485
        mean[1] = 0.456
        mean[2] = 0.406
        mean = Variable(mean)
        mean = mean.resize(1, 3, 1, 1).cuda()

        std = torch.FloatTensor(3)
        std[0] = 0.229
        std[1] = 0.224
        std[2] = 0.225
        std = Variable(std)
        std = std.resize(1, 3, 1, 1).cuda()

        """

        inputs_arranged = inputs.view(inputs.size(0), inputs.size(1), -1)
        std_input, mean_input = torch.std_mean(inputs_arranged, dim=2)
        mean_input = mean_input.sum(0)
        mean_input = Variable(mean_input).resize(1, 3, 1, 1).cuda()
        std_input = std_input.sum(0)
        std_input = Variable(std_input).resize(1, 3, 1, 1).cuda()

        #Source image
        targets_arranged = targets_1.view(targets_1.size(0), targets_1.size(1), -1)
        std_targets, mean_targets = torch.std_mean(targets_arranged, dim=2)
        mean_targets = mean_targets.sum(0)
        mean_targets = Variable(mean_targets).resize(1, 3, 1, 1).cuda()
        std_targets = std_targets.sum(0)
        std_targets = Variable(std_targets).resize(1, 3, 1, 1).cuda()
        #Target image
        targets_arranged_2 = targets_2.view(targets_2.size(0), targets_2.size(1), -1)
        std_targets_2, mean_targets_2 = torch.std_mean(targets_arranged_2, dim=2)
        mean_targets_2 = mean_targets_2.sum(0)
        mean_targets_2 = Variable(mean_targets_2).resize(1, 3, 1, 1).cuda()
        std_targets_2 = std_targets_2.sum(0)
        std_targets_2 = Variable(std_targets_2).resize(1, 3, 1, 1).cuda()






        fake_p2_norm = (inputs + 1)/2 # [-1, 1] => [0, 1]
        fake_p2_norm = (fake_p2_norm - mean_input)/std_input

        #Source image
        input_p2_norm = (targets_1 + 1)/2 # [-1, 1] => [0, 1]
        input_p2_norm = (input_p2_norm - mean_targets)/std_targets
        #Target image
        input_p2_norm_2 = (targets_2 + 1)/2 # [-1, 1] => [0, 1]
        input_p2_norm_2 = (input_p2_norm_2 - mean_targets_2)/std_targets_2

        fake_p2_norm_1 = self.vgg_submodel(fake_p2_norm)
        input_p2_norm = self.vgg_submodel(input_p2_norm)

        fake_p2_norm_2 = self.vgg_submodel_2(fake_p2_norm)
        input_p2_norm_2 = self.vgg_submodel_2(input_p2_norm_2)

        input_p2_norm_no_grad = input_p2_norm.detach()
        input_p2_norm_no_grad_2 = input_p2_norm_2.detach()

        if self.percep_is_l1 == 1:
            # use l1 for perceptual loss
            loss_perceptual_source = F.l1_loss(fake_p2_norm_1, input_p2_norm_no_grad) * self.lambda_perceptual
            #Error of different input size
            loss_perceptual_target = F.l1_loss(fake_p2_norm_2, input_p2_norm_no_grad_2) * self.lambda_perceptual
        else:
            # use l2 for perceptual loss
            loss_perceptual_source = F.mse_loss(fake_p2_norm_1, input_p2_norm_no_grad) * self.lambda_perceptual
            loss_perceptual_target = F.mse_loss(fake_p2_norm_2, input_p2_norm_no_grad_2) * self.lambda_perceptual

        #loss = loss_l1 + loss_perceptual_source
        loss = loss_l1 + loss_perceptual_source + loss_perceptual_target

        #return loss, loss_l1, loss_perceptual_source
        return loss, loss_l1, loss_perceptual_source, loss_perceptual_target 

