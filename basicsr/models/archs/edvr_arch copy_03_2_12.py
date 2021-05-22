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

        # pcd and tsa module
        self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
        self.conv1x1_1 = nn.Conv2d(num_feat*(num_frame - 1), num_feat, 1, 1)
        self.conv1x1_2 = nn.Conv2d(num_feat, num_feat, 1, 1)
        # reconstruction
        # self.reconstruction = make_layer(
        #     ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv3 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upconvs = [self.upconv3, self.upconv2, self.upconv1]
        # self.upconv3 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upconv_final1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv_final2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.downsample = nn.MaxPool2d(8)

        # kaiming_normal_
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def upsampleBlock(self, feat_qk, b, t, h, w):
        # print(feat_qk.shape) # qk shape: batch*num_frames 32 2*h 2*w
        dh = int(h/8)
        dw = int(w/8)
        
        # downsample feat_qk 8x
        # TODO: 3*3 conv downsample
        down_x8_qk = self.downsample(feat_qk)
        # print('feat_qk: ', feat_qk.shape) b*num_frames 64 h w
        # print(down_x8_qk.shape) # batch*num_frames 64 dh dw

        # reshape feat_qk to split ref_frame&other frames
        feat_qk = feat_qk.view(b, t, -1, h, w)
        down_x8_qk = down_x8_qk.view(b, t, -1, dh, dw)
        # extract q/k of ref_frame and other frames, then repeat ref_frame
        split_fk1, split_fq, split_fk2= torch.split(feat_qk, [int(t/2),1,int(t/2)], dim=1)
        fk = torch.cat((split_fk1, split_fk2), 1).view(b*(t-1), -1, h, w).float()
        fq = split_fq.repeat([1,(t-1),1,1,1]).view(b*(t-1), -1, h, w).float()
        # print(fk.shape) # batch*(num_frames-1) dim h w
        # print(fq.shape) # batch*(num_frames-1) dim h w
        
        # corr softmax of qxk
        corr_qxk = CorrBlock(fq, fk, num_levels=self.num_levels, radius=self.corr_radius)
        
        # prepare for recurrent upsampling
        split_dfk1, split_dfq, split_dfk2= torch.split(down_x8_qk, [int(t/2),1,int(t/2)], dim=1)
        dfk = torch.cat((split_dfk1, split_dfk2), 1).view(b*(t-1), -1, dh, dw).float()
        dfq = split_dfq.repeat([1,(t-1),1,1,1]).view(b*(t-1), -1, dh, dw).float()
        # print('sdfq: ', split_dfq.shape) # batch 1 dim dh dw
        # main block
        for i in range(self.num_levels-1, -1, -1):
            # update dfq
            soft1x2 = corr_qxk.corr_pyramid_1x2[i] # softmax 1x2
            # print('199 soft: ', soft1x2.shape)
            # print('199 dfk: ', dfk.shape) # batch*(t-1), dim, h, w
            _, inter_channel, dfk_h, dfk_w = dfk.shape
            sdfk = dfk.view(b*(t-1), inter_channel, dfk_h*dfk_w)
            sdfk = sdfk.permute(0,2,1)
            sdfk = torch.matmul(soft1x2, sdfk)
            sdfk = sdfk.permute(0, 2, 1).contiguous().view(b, (t-1)*inter_channel, dfk_h, dfk_w)
            sdfk = self.lrelu(self.conv1x1_1(sdfk))
            dfq_2 = split_dfq.view(b, -1, dfk_h, dfk_w)
            dfq_2 = dfq_2 + sdfk
            # print(sdfk.shape, dfq_2.shape) # batch dim h w

            # update dfk
            soft2x1 = corr_qxk.corr_pyramid_2x1[i] # softmax 2x1
            dfq = dfq.view(b*(t-1), inter_channel, dfk_h*dfk_w)
            dfq = dfq.permute(0,2,1)
            sdfq = torch.matmul(soft2x1, dfq) # batch h*w dim
            sdfq = sdfq.permute(0, 2, 1).contiguous().view(b*(t-1), inter_channel, dfk_h, dfk_w)
            sdfq = self.lrelu(self.conv1x1_2(sdfq))
            dfk_2 = dfk + sdfq
            # print(dfk.shape, sdfq.shape) # batch*(num_frames-1) dim dh dw

            # upsample & prepare for next iter
            if i == 0:
                split_dfq = self.lrelu(self.conv4(dfq_2))
                dfk = self.lrelu(self.conv4(dfk_2))
                dfq = split_dfq.repeat([1,(t-1),1,1,1]).view(b*(t-1), -1, dh, dw).float()
            else:
                split_dfq = self.lrelu(self.pixel_shuffle(self.upconvs[i-1](dfq_2)))
                dfk = self.lrelu(self.pixel_shuffle(self.upconvs[i-1](dfk_2)))
                dfq = split_dfq.repeat([1,(t-1),1,1,1]).view(b*(t-1), -1, dh, dw).float()
        return split_dfq, dfk, dfq
    
    def pre_recurrent(self, split_dfq, dfk, b, t, h, w):
        dfk = dfk.view(b, (t-1), -1, h, w)
        split_dfk1, split_dfk2= torch.split(dfk, [int(t/2),int(t/2)], dim=1)
        split_dfq = split_dfq.unsqueeze(1)
        # print(dfk.shape, split_dfq.shape)
        # assert(0)
        recur_feat = torch.cat((split_dfk1, split_dfq, split_dfk2), dim=1)
        recur_feat = recur_feat.view(b*t, -1, h, w)
        return recur_feat

    def forward(self, x):
        b, t, c, h, w = x.size()

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()
        # print('x_center: ', x_center.shape) # 2 3 32 24
        # f_v = feature_extractor_v(x.view(-1,c,h,w))
        # extract qkv for each frame
        feat_qk = self.feature_extractor_qk(x.view(-1, c, h, w))
        
        h = 2*h
        w = 2*w

        split_dfq1, dfk, dfq = self.upsampleBlock(feat_qk, b, t, h, w)
        # feat_qk1 = self.pre_recurrent(split_dfq1, dfk, b, t, h, w)
        # split_dfq2, dfk, dfq = self.upsampleBlock(feat_qk1, b, t, h, w)
        # feat_qk3 = self.pre_recurrent(split_dfq2, dfk, b, t, h, w)
        # split_dfq3, dfk, dfq = self.upsampleBlock(feat_qk3, b, t, h, w)
        
        out = self.lrelu(self.pixel_shuffle(self.upconv_final1(split_dfq1)))
        out = self.lrelu(self.pixel_shuffle(self.upconv_final2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        if self.hr_in:
            base = x_center
        else:
            base = F.interpolate(
                x_center, scale_factor=8, mode='bicubic', align_corners=False)
        out += base
        return out, None, None