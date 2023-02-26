import torch.nn as nn
import clip
from config import cfg
from layers import *
from torch.autograd import Variable

# baseline:stackGAN-v-2(이하 stackGAN)

#######################################################################################################
# DO NOT CHANGE  
class GENERATOR(nn.Module):
#######################################################################################################
    def __init__(self):
        super(GENERATOR, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = Get_init_code(self.gf_dim * 8)
            self.img_net1 = Get_Image(self.gf_dim) # channel
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = Gan_middle(self.gf_dim)
            self.img_net2 = Get_Image(self.gf_dim // 2)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = Gan_middle(self.gf_dim // 2)
            self.img_net3 = Get_Image(self.gf_dim // 4)
        if cfg.TREE.BRANCH_NUM > 3:
            self.h_net4 = Gan_middle(self.gf_dim // 4, num_residual=1)
            self.img_net4 = Get_Image(self.gf_dim // 8)

        
    def forward(self, z_code, text_embedding, z2w):
        """
        z_code: batch x cfg.GAN.Z_DIM
        sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
        return: generated image
        """

        if cfg.CA:
            c_code, mu, logvar = self.ca_net(text_embedding) # with CA
        else:
            c_code, mu, logvar = text_embedding, None, None # without CA
        fake_imgs = []
        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code, z2w)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3 = self.h_net3(h_code2, c_code)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
        if cfg.TREE.BRANCH_NUM > 3:
            h_code4 = self.h_net4(h_code3, c_code)
            fake_img4 = self.img_net4(h_code4)
            fake_imgs.append(fake_img4)

        return fake_imgs, mu, logvar

# first part for stackGAN
class Get_init_code(nn.Module):
    def __init__(self, mul_channel):
        super(Get_init_code, self).__init__()
        self.mul_channel = mul_channel
        self.in_dim = cfg.GAN.Z_DIM + cfg.GAN.EMBEDDING_DIM

        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.mul_channel*4*4*2, bias=False),
            nn.BatchNorm1d(self.mul_channel*4*4*2),
            GLU()
        )

        self.new_fc = nn.Sequential(
            nn.Linear(self.in_dim, self.mul_channel * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.mul_channel * 4 * 4 * 2),
            GLU(),
            nn.Linear(self.mul_channel*4*4*2, self.mul_channel * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.mul_channel * 4 * 4 * 2),
            GLU(),
            nn.Linear(self.in_dim, self.mul_channel * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.mul_channel * 4 * 4 * 2),
            GLU()
        )

        self.upsample1 = upBlock(self.mul_channel, self.mul_channel//2)
        self.upsample2 = upBlock(self.mul_channel//2, self.mul_channel//4)
        self.upsample3 = upBlock(self.mul_channel//4, self.mul_channel//8)

    def forward(self, z, c, z2w):

        in_code = torch.cat((c,z), dim=1)

        if z2w:
            out = self.new_fc(in_code)
        else:
            out = self.fc(in_code)
        out = out.view(-1, self.mul_channel, 4, 4)
        out = self.upsample1(out)
        out = self.upsample2(out)
        out = self.upsample3(out)

        return out

class Gan_middle(nn.Module):
    def __init__(self, mul_channel, num_residual=cfg.GAN.R_NUM ):
        super(Gan_middle, self).__init__()
        self.mul_channel = mul_channel
        self.emb_dim = cfg.GAN.EMBEDDING_DIM
        self.num_residual = num_residual

        self.jointConv = Block3x3_relu(self.mul_channel + self.emb_dim, self.mul_channel)
        self.residual = self._make_layer(ResBlock, self.mul_channel)
        self.upsample = upBlock(self.mul_channel, self.mul_channel // 2)

    def _make_layer(self, block, channel_num):
        layers = list()
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(self, h, c):
        s_size = h.size(2)
        c = c.view(-1, self.emb_dim, 1, 1)
        c = c.repeat(1, 1, s_size, s_size)

        h_c = torch.cat((c, h), dim=1)

        out = self.jointConv(h_c)
        out = self.residual(out)
        out = self.upsample(out)

        return out

class Get_Image(nn.Module):
    def __init__(self, mul_channel):
        super(Get_Image, self).__init__()
        self.mul_channel = mul_channel
        self.img = nn.Sequential(
            conv3x3(self.mul_channel, 3),
            nn.Tanh()
        )

    def forward(self, h):
        img = self.img(h)
        return img

# ablation study
class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

#######################################################################################################
# DO NOT CHANGE 
class DISCRIMINATOR(nn.Module):
#######################################################################################################
    def __init__(self,):
        super(DISCRIMINATOR, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x_var):
        '''
        x_var: real or generated images
        return: probability whether the x_var is real or fake
        '''
        pass

class D_NET32(nn.Module):
    def __init__(self):
        super(D_NET32, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s8 = encode_image_by_8times(ndf) # 32*32 -> 4*4

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s8(x_var)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)

        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]

class D_NET64(nn.Module):
    def __init__(self):
        super(D_NET64, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s8 = encode_image_by_8times(ndf)
        self.img_down = downBlock(ndf*8, ndf*16)
        self.img_leakRelu = Block3x3_leakRelu(ndf*16, ndf*8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s8(x_var)
        x_code = self.img_down(x_code)
        x_code = self.img_leakRelu(x_code)

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)

        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]

class D_NET128(nn.Module):
    def __init__(self):
        super(D_NET128, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s8 = encode_image_by_8times(ndf)
        self.img_down1 = downBlock(ndf*8, ndf*16)
        self.img_down2 = downBlock(ndf*16, ndf*32)
        self.img_leakRelu1 = Block3x3_leakRelu(ndf*32, ndf*16)
        self.img_leakRelu2 = Block3x3_leakRelu(ndf*16, ndf*8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s8(x_var)
        x_code = self.img_down1(x_code)
        x_code = self.img_down2(x_code)
        x_code = self.img_leakRelu1(x_code)
        x_code = self.img_leakRelu2(x_code)

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)

        out_uncond = self.uncond_logits(x_code)
        return [output.view(-1), out_uncond.view(-1)]
