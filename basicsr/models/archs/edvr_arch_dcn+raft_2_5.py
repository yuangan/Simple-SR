import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.models.archs.arch_util import (DCNv2Pack, ResidualBlockNoBN,
                                            make_layer, BasicEncoder, coords_grid)
from basicsr.models.archs.update import BasicUpdateBlock
from basicsr.models.archs.corr import CorrBlock
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class warp(torch.nn.Module):
    def __init__(self, h, w, cuda_flag):
        super(warp, self).__init__()
        self.height = h
        self.width = w
        if cuda_flag:
            self.addterm = self.init_addterm().cuda()
        else:
            self.addterm = self.init_addterm()

    def init_addterm(self):
        n = torch.FloatTensor(list(range(self.width)))
        horizontal_term = n.expand((1, 1, self.height, self.width))  # 第一个1是batch size
        n = torch.FloatTensor(list(range(self.height)))
        vertical_term = n.expand((1, 1, self.width, self.height)).permute(0, 1, 3, 2)
        addterm = torch.cat((horizontal_term, vertical_term), dim=1)
        return addterm

    def forward(self, frame, flow):
        """
        :param frame: frame.shape (batch_size=1, n_channels=3, width=256, height=448)
        :param flow: flow.shape (batch_size=1, n_channels=2, width=256, height=448)
        :return: reference_frame: warped frame
        """
        if True:
            flow = flow + self.addterm
        else:
            self.addterm = self.init_addterm()
            flow = flow + self.addterm

        horizontal_flow = flow[0, 0, :, :].expand(1, 1, self.height, self.width)  # 第一个0是batch size
        vertical_flow = flow[0, 1, :, :].expand(1, 1, self.height, self.width)

        horizontal_flow = horizontal_flow * 2 / (self.width - 1) - 1
        vertical_flow = vertical_flow * 2 / (self.height - 1) - 1
        flow = torch.cat((horizontal_flow, vertical_flow), dim=1)
        flow = flow.permute(0, 2, 3, 1)
        reference_frame = torch.nn.functional.grid_sample(frame, flow)
        return reference_frame

class EDVR(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.
    Paper:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=2,
                 hr_in=False,
                 with_predeblur=False,
                 with_tsa=True,
                 dropout=0.0):
        super(EDVR, self).__init__()


        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.dropout = dropout

        # RAFT Parameter
        self.corr_radius = 4
        self.num_levels = 4
        self.mixed_precision = False
        self.iters = 6

        # extrat v features
        self.feature_extractor_v = make_layer(
            ResidualBlockNoBN, num_extract_block, num_feat=num_feat)

        # RAFT Network
        # extrat qk/c features
        self.feature_extractor_qk = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)
        self.feature_extractor_c = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=self.dropout)
        # update flow
        self.update_block = BasicUpdateBlock(self.num_levels, self.corr_radius, hidden_dim=hdim)

        # pcd and tsa module
        self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        self.reconstruction = make_layer(
            ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.upconv3 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        


        # kaiming_normal_
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, T, C, H, W = img.shape
        coords0 = coords_grid(N*(T-1), H, W).to(img.device)
        coords1 = coords_grid(N*(T-1), H, W).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def forward(self, x):
        b, t, c, h, w = x.size()

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()
        print('x_center: ', x_center.shape)
        # f_v = feature_extractor_v(x.view(-1,c,h,w))
        # extract qkv for each frame
        feat_qk = self.feature_extractor_qk(x.view(-1, c, h, w))
        feat_c = self.feature_extractor_c(x_center.view(-1, c, h, w))
        # print(feat_qk.shape) # qk shape: 2*7 256 32 24
        # print(feat_v.shape) # v shape: 2*7 256 32 24

        # reshape feat_qk to split ref_frame&other frames
        feat_qk = feat_qk.view(b, t, -1, h, w)
        feat_c = feat_c.view(b, 1, -1, h, w)

        # extract q/k of ref_frame and other frames, then repeat ref_frame
        split_fk1, split_fq, split_fk2= torch.split(feat_qk, [int(t/2),1,int(t/2)], dim=1)
        fk = torch.cat((split_fk1, split_fk2), 1).view(b*(t-1), -1, h, w).float()
        fq = split_fq.repeat([1,(t-1),1,1,1]).view(b*(t-1), -1, h, w).float()
        # print(fk.shape) # 2*6, 256, 32, 24
        # print(fq.shape) # 2*6, 256, 32, 24

        # corr of kxq, optical flow k->q
        corr_fn = CorrBlock(fk, fq, num_levels=self.num_levels, radius=self.corr_radius)

        # split v into net&inp
        with autocast(enabled=self.mixed_precision):
        # cnet = self.cnet(image1)
            feat_c = feat_c.repeat([1,(t-1),1,1,1]).view(b*(t-1), -1, h, w)
            net, inp = torch.split(feat_c, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(x)
        flow_predictions = []
        for itr in range(self.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            flow_up = coords1 - coords0
            print(flow_up.shape) # 2*6 2 32 24
            # upsample predictions
            # if up_mask is None:
            #     flow_up = upflow8(coords1 - coords0)
            # else:
            #     flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        # if test_mode:
        #     return coords1 - coords0, flow_up
        
        # assert(0)
        # return flow_predictions

        
        aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv3(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        if self.hr_in:
            base = x_center
        else:
            base = F.interpolate(
                x_center, scale_factor=8, mode='bilinear', align_corners=False)
        out += base
        return out, None, None