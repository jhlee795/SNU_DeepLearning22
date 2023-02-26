import torch
import clip
import torch.nn as nn
from config import cfg

class CLIPLoss(nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, _ = clip.load('ViT-B/32', device = 'cuda')
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool64 = torch.nn.AvgPool2d(kernel_size=2)
        self.avg_pool128 = torch.nn.AvgPool2d(kernel_size=4)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, image, text):
        image = self.upsample(image)
        image = self.avg_pool128(image)
        result = self.model(image,text)
        result = torch.diag(result[0], 0)
        result = torch.mean(result)/100
        similarity = 1 - result

        return similarity

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """
    Returns cosine similarity between x1 and x2, computed along dim
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def compute_mean_covariance(img):
    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width

    # batch_size * channel_num * 1 * 1
    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

    # batch_size * channel_num * num_pixels
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    # batch_size * num_pixels * channel_num
    img_hat_transpose = img_hat.transpose(1, 2)
    # batch_size * channel_num * channel_num
    covariance = torch.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
