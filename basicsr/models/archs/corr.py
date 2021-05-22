import torch
import torch.nn.functional as F
from basicsr.models.archs.utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass

 
class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid_1x2 = []
        self.corr_pyramid_2x1 = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        # print('batch, h1, w1, dim, h2, w2: ', corr.shape) # 12, 32, 24, 1, 32, 24
        corr1 = corr.reshape(batch*h1*w1, dim, h2, w2)
        corr1x2 = corr1.reshape(batch, h1*w1, h2*w2)
        corr2x1 = corr1x2.permute(0, 2, 1)# batch, tmp_h1*tmp_w1, tmp_h2*tmp_w2
        self.corr_pyramid_1x2.append(corr1x2)
        self.corr_pyramid_2x1.append(corr2x1)
        for i in range(self.num_levels-1):
            # print(corr1.shape)
            # 2d pooling h2 w2
            corr1 = F.avg_pool2d(corr1, 2, stride=2)
            
            # transpose corr in order to pooling h1 w1
            _, _, tmp_h, tmp_w = corr1.shape
            corr1 = corr1.reshape(batch, tmp_h*2, tmp_w*2, dim, tmp_h*tmp_w) 
            corr2 = corr1.permute(0, 4, 3, 1, 2) # batch, tmp_h*tmp_w, dim, h1, w1
            corr2 = corr2.reshape(batch*tmp_h*tmp_w, dim, tmp_h*2, tmp_w*2)

            # 2d pooling h1 h2 
            # TODO: 4d pooling may be more efficient
            corr2 = F.avg_pool2d(corr2, 2, stride=2) # batch*tmp_h2*tmp_w2,dim,tmp_h1,tmp_w1
            # print('pooling corr: ', corr.shape)
            corr2x1 = corr2.reshape(batch, tmp_h*tmp_w, tmp_h*tmp_w) # batch, tmp_h2*tmp_w2, tmp_h1*tmp_w1
            corr1x2 = corr2x1.permute(0, 2, 1)# batch, tmp_h1*tmp_w1, tmp_h2*tmp_w2
            softmax_2x1 = F.softmax(corr2x1, dim=-1)
            softmax_1x2 = F.softmax(corr1x2, dim=-1)
            self.corr_pyramid_2x1.append(softmax_2x1)
            self.corr_pyramid_1x2.append(softmax_1x2)

            # prepare for next iter
            corr1 = corr1x2.reshape(batch*tmp_h*tmp_w, dim, tmp_h, tmp_w)

    # def __call__(self, coords):
    #     r = self.radius
    #     coords = coords.permute(0, 2, 3, 1)
    #     batch, h1, w1, _ = coords.shape

    #     out_pyramid = []
    #     for i in range(self.num_levels):
    #         corr = self.corr_pyramid[i]
    #         dx = torch.linspace(-r, r, 2*r+1)
    #         dy = torch.linspace(-r, r, 2*r+1)
    #         delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

    #         centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
    #         delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
    #         coords_lvl = centroid_lvl + delta_lvl

    #         corr = bilinear_sampler(corr, coords_lvl)
    #         corr = corr.view(batch, h1, w1, -1)
    #         out_pyramid.append(corr)

    #     out = torch.cat(out_pyramid, dim=-1)
    #     return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)
        # print('fmap1: ', fmap1.shape)# b*(t-1) dim 64*48 
        # print('fmap2: ', fmap2.shape)
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
