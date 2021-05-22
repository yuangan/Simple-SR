import numpy as np
import cv2

root = 'flow_save_160/'

def get_mask(mask, pic_index, offset_index):
    hsv = np.zeros((160, 160, 3), np.uint8)
    print('mask: ', mask.shape)
    mask = mask [0,pic_index,offset_index]
    mask = mask.mean(axis=0)
    # mask = mask.mean(axis=0)
    mag, ang = cv2.cartToPolar(mask,mask)
    hsv[...,0] = ang*180/np.pi/2
   # hsv[...,0] = cv2.normalize(mask,None,0,255,cv2.NORM_MINMAX)
   # hsv[...,1] = cv2.normalize(mask,None,0,255,cv2.NORM_MINMAX)
    hsv[...,1] = cv2.normalize(mask,None,0,255,cv2.NORM_MINMAX)
    hsv[...,2] = 255
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imwrite(root+'mask.png',bgr)
def get_flow(flow, pic_index, offset_index):
    print('flow: ', flow.shape)
    flow = flow [0, pic_index, offset_index]
    #flow = flow.mean(axis=0)
    flow = flow.reshape(8,18,160,160)
    flow = flow.mean(axis=0)
#     flow_x = flow[::2,:,:].mean(axis=0)
#     flow_y = flow[1::2,:,:].mean(axis=0)
    flow_x = flow[0]
    flow_y = flow[1]
    h, w = flow_x.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow_y,flow_x)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # flownet是将V赋值为255, 此函数遵循flownet，饱和度S代表像素位移的大小，亮度都为最大，便于观看
    # 也有的光流可视化讲s赋值为255，亮度代表像素位移的大小，整个图片会很暗，很少这样用
    hsv[...,2] = 255
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imwrite(root+'flow.png',bgr)
    print(flow_x.mean(),flow_y.mean())
def get_gt(flow,pic_index):
    flow = flow[pic_index]

#     flow_x = flow[::2,:,:].mean(axis=0)
#     flow_y = flow[1::2,:,:].mean(axis=0)
    flow_x = flow[...,0].astype('float')
    flow_y = flow[...,1].astype('float')
    print(flow_x.shape)
    print(flow_y.shape)
    flow_x[np.abs(flow_x)<5] = 0
    flow_y[np.abs(flow_y)<5] = 0
    h, w = flow_x.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    
    mag, ang = cv2.cartToPolar(flow_y,flow_x)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # flownet是将V赋值为255, 此函数遵循flownet，饱和度S代表像素位移的大小，亮度都为最大，便于观看
    # 也有的光流可视化讲s赋值为255，亮度代表像素位移的大小，整个图片会很暗，很少这样用
    hsv[...,2] = 255
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imwrite(root+'gt.png',bgr)
    print(flow_x.mean(),flow_y.mean())

offset_index = 0
pic_index = 0
flow = np.load(root+'offset.npy')
mask = np.load(root+'mask.npy')
#gt = np.load('/data2/wei/vox-160/GT/id06060/lwF1jB7DnMo/flow_160.npy')

#gt = np.load('/data2/wei/vox-160/GT/id00060/jKqs7j3iRxo/flow_160.npy')

gt = np.load('/data2/wei/vox-160/GT/id06060/2GQlRxUF_7c/flow_160.npy')
#gt = np.load('/data2/wei/vox-160/GT/id00741/8XrTdMGunzg/flow_160.npy')
get_flow(flow, pic_index,offset_index)
get_gt(gt/2.0, pic_index)
get_mask(mask, pic_index, offset_index)