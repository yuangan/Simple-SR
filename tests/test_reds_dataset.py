import math
import mmcv
import torchvision.utils

from basicsr.data import create_dataloader, create_dataset


def main(mode='folder'):
    """Test reds dataset.

    Args:
        mode: There are two modes: 'lmdb', 'folder'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'REDS'
    opt['type'] = 'REDSDataset'
    if mode == 'folder':
        opt['dataroot_gt'] = 'datasets/REDS/train_sharp'
        opt['dataroot_lq'] = 'datasets/REDS/train_sharp_bicubic'
        opt['dataroot_flow'] = None
        opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_REDS_GT.txt'
        opt['io_backend'] = dict(type='disk')
    elif mode == 'lmdb':
        opt['dataroot_gt'] = 'datasets/REDS/train_sharp_with_val.lmdb'
        opt['dataroot_lq'] = 'datasets/REDS/train_sharp_bicubic_with_val.lmdb'
        opt['dataroot_flow'] = None
        opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_REDS_GT.txt'
        opt['io_backend'] = dict(type='lmdb')

    opt['val_partition'] = 'REDS4'
    opt['num_frame'] = 5
    opt['gt_size'] = 256
    opt['interval_list'] = [1]
    opt['random_reverse'] = True
    opt['use_flip'] = True
    opt['use_rot'] = True

    opt['use_shuffle'] = True
    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 16
    opt['scale'] = 4

    opt['dataset_enlarge_ratio'] = 1

    mmcv.mkdir_or_exist('tmp')

    dataset = create_dataset(opt)
    data_loader = create_dataloader(
        dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 5:
            break
        print(i)

        lq = data['lq']
        gt = data['gt']
        key = data['key']
        print(key)
        for j in range(opt['num_frame']):
            torchvision.utils.save_image(
                lq[:, j, :, :, :],
                f'tmp/lq_{i:03d}_frame{j}.png',
                nrow=nrow,
                padding=padding,
                normalize=False)
        torchvision.utils.save_image(
            gt,
            f'tmp/gt_{i:03d}.png',
            nrow=nrow,
            padding=padding,
            normalize=False)


if __name__ == '__main__':
    main()
