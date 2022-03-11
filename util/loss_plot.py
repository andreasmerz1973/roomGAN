import numpy as np
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def smooth(loss):
    for i in range(20, len(loss)):
        loss[i-20] = np.mean(loss[i-20:i])
    for i in range(len(loss) - 20, len(loss)):
        loss[i] = loss[len(loss) - 21]
    return loss

use_pairl1 = True
use_originl1 = False
use_dpb = True
use_dpp = True
use_gan = True
use_perceptual = False
use_cx = True

model_name = 'scannet_clean'


if __name__ == '__main__':

    with open(os.path.join('checkpoints', model_name, 'loss_log.txt'), 'r') as file1:
        lines = file1.read()

    if not os.path.exists(os.path.join('checkpoints', model_name, 'loss_plots')):
        os.mkdir(os.path.join('checkpoints', model_name, 'loss_plots'))


    if use_pairl1:
        pairl1_loss = list(map(float, re.findall("pair_L1loss: ([^ ]+)", lines)))
        
        iterations = np.linspace(0, len(pairl1_loss)*100, len(pairl1_loss))
        inter_iterations = np.linspace(0, len(pairl1_loss)*100, len(pairl1_loss)*1000)

        plt.title("Pair L1 Loss")
        plt.xlabel("Iterations")
        plt.ylim(0, max(pairl1_loss) + 1)
        plt.plot(iterations, pairl1_loss, '--', label = 'original')
        plt.plot(iterations, smooth(pairl1_loss), '-', label = 'smooth')
        plt.savefig(os.path.join('checkpoints', model_name, 'loss_plots', 'pairl1_loss.png'), bbox_inches='tight')
        plt.clf()

    if use_originl1:
        originl1_loss = list(map(float, re.findall("origin_L1: ([^ ]+)", lines)))
        plt.title("Origin L1 Loss")
        plt.xlabel("Iterations")
        plt.ylim(0, max(originl1_loss) + 1)
        plt.plot(iterations, originl1_loss)
        plt.plot(iterations, smooth(originl1_loss))
        plt.savefig(os.path.join('checkpoints', model_name, 'loss_plots', 'originl1_loss.png'), bbox_inches='tight')
        plt.clf()

    if use_dpb:
        dpb_loss = list(map(float, re.findall("D_PB: ([^ ]+)", lines)))
        plt.title("D_PB Loss")
        plt.xlabel("Iterations")
        plt.ylim(0, max(dpb_loss) + 1)
        plt.plot(iterations, dpb_loss)
        plt.plot(iterations, smooth(dpb_loss))
        plt.savefig(os.path.join('checkpoints', model_name, 'loss_plots', 'dpb_loss.png'), bbox_inches='tight')
        plt.clf()

    if use_dpb:
        dpp_loss = list(map(float, re.findall("D_PP: ([^ ]+)", lines)))
        plt.title("D_PP Loss")
        plt.xlabel("Iterations")
        plt.ylim(0, max(dpb_loss) + 1)
        plt.plot(iterations, dpb_loss)
        plt.plot(iterations, smooth(dpb_loss))
        plt.savefig(os.path.join('checkpoints', model_name, 'loss_plots', 'dpp_loss.png'), bbox_inches='tight')
        plt.clf()

    if use_gan:
        gan_loss = list(map(float, re.findall("pair_GANloss: ([^ ]+)", lines)))
        plt.title("GAN Loss")
        plt.xlabel("Iterations")
        plt.ylim(0, max(gan_loss) + 1)
        plt.plot(iterations, gan_loss)
        plt.plot(iterations, smooth(gan_loss))
        plt.savefig(os.path.join('checkpoints', model_name, 'loss_plots', 'gan_loss.png'), bbox_inches='tight')
        plt.clf()

    if use_perceptual:
        perceptual_loss = list(map(float, re.findall("perceptual: ([^ ]+)", lines)))
        perceptual_loss_target = list(map(float, re.findall("perceptual_target: ([^ ]+)", lines)))

        plt.title("Perceptual Loss")
        plt.xlabel("Iterations")
        plt.ylim(0, max(perceptual_loss) + 1)
        plt.plot(iterations, perceptual_loss)
        plt.plot(iterations, smooth(perceptual_loss))
        plt.savefig(os.path.join('checkpoints', model_name, 'loss_plots', 'perceptual_loss.png'), bbox_inches='tight')
        plt.clf()
        plt.title("Perceptual Loss (Target)")
        plt.xlabel("Iterations")
        plt.ylim(0, max(perceptual_loss_target) + 1)
        plt.plot(iterations, perceptual_loss_target)
        plt.plot(iterations, smooth(perceptual_loss_target))
        plt.savefig(os.path.join('checkpoints', model_name, 'loss_plots', 'perceptual_loss_target.png'), bbox_inches='tight')
        plt.clf()
    
    if use_cx:
        cx_loss = list(map(float, re.findall("CXLoss: ([^ ]+)", lines)))
        plt.title("Contextual Loss")
        plt.xlabel("Iterations")
        plt.ylim(0, max(cx_loss) + 1)
        plt.plot(iterations, cx_loss)
        plt.plot(iterations, smooth(cx_loss), '-')
        plt.savefig(os.path.join('checkpoints', model_name, 'loss_plots', 'cx_loss.png'), bbox_inches='tight')
        plt.clf()

