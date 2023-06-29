import argparse
import os
import platform
import torch
import torch.nn.functional as F
import yaml
import json
import numpy as np
# import cv2
import natsort
from tensorboardX import SummaryWriter

#from convViT import ConvolutionalVisionTransformer
from convDownModule import Encoder
from loader_simple import get_loader
from autoencvec import AutoEncoder, init_weights


parser = argparse.ArgumentParser(description='ProgettoX')
parser.add_argument('--cfg', type=str, #required=True,
                    help='config file path', default='./configs/config13-224.yaml')
parser.add_argument('--indir', dest='inputpath',
                    help='Path to input data', default="/capital/datasets/BEHAVE_updated")
parser.add_argument('--epochs', dest='epochs', type=int,
                    help='number epochs to train', default=100)
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--video', dest='video', type=bool,
                    help='Input data is videofile', default=False)
parser.add_argument('--mb', type=int, default=16,
                    help='minibatch size')
parser.add_argument('--item-type', type=str, default='mix',
                    help='type of item to add to minibatch: sequential or mixed')
parser.add_argument('--dev-batch', type=int, default=1,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--pretrained', type=bool, help='Loading weights from file', default=False)
parser.add_argument('--check', type=str, default='./checkpoint', help='directory to save parameters')
parser.add_argument('--check-load', type=str, default='./checkpoint_load', help='directory to save parameters')
parser.add_argument('--sigma', type=float, default=0.5)  # 1.0 is working version
parser.add_argument('--stat', type=str, default='./stat', help='directory to save statistics')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')


args = parser.parse_args()

if not os.path.exists(args.stat):
    os.makedirs(args.stat)

writer = SummaryWriter(args.stat)

with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader) 

if platform.system() == 'Windows':
    args.sp = True
else:
    args.sp = False

if not os.path.exists(args.check):
    os.makedirs(f'{args.check}/conv')
    os.makedirs(f'{args.check}/vae')

args.images = args.video==False

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.dev_batch = args.dev_batch * len(args.gpus)

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

def check_integrity(jroot, root, jsons, videos):
    jsns, vds = [], []
    jsons = natsort.natsorted(jsons)
    videos = natsort.natsorted(videos)
    max_dist = []
    for jsn, vdo in zip(jsons, videos):

        with open(os.path.join(jroot, jsn), "r") as read_file:
            results = json.load(read_file)
            max_dist.append(np.array(results).max())
        jlength = len(results)   

        stream = cv2.VideoCapture(os.path.join(root, vdo))
        assert stream.isOpened(), 'Cannot capture source'
        vlength = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        stream.release() 

        if jlength == vlength and jlength > args.mb:
            jsns.append(jsn)
            vds.append(vdo)
            #print(f'For files {jsn}  {vlength} == {jlength}')
        else:
            print(f'Warning: number of frames in the video {vdo} is not compatible ', 
            f'with the number of frames in json ({vlength} / {jlength}) ',
            f'this pair will be excluded from dataset')
            pass
   
    assert len(jsns) == len(vds), f'Two sequece have different sizes'
    print(f'Final dataset contains {len(jsns)} entries from initial {len(jsons)}')
    return jsns, vds, np.max(max_dist)


def create_separete_jsosns(path, files):
    for count, vec in enumerate(files):
        with open(os.path.join(path, str(count).zfill(5)+'.json'), 'w') as json_file:
            json_file.write(json.dumps(vec))


def check_input(inputpath, inputlist='', inputimg=''):       
    # for video
    if args.video:
        max_dists = []
        video_dataset = dict({'videos': [], 'jsons': []})
        for root, dirs, files in os.walk(inputpath):
            try: 
                name, ext = os.path.splitext(files[0])
            except:
                continue
            else:
                if ext not in ['.mp4', '.avi', '.mov']:
                    print('The files in folder are not videos!')
                    continue

            core_name = root[:-7]
            jsons_dir = os.path.join(core_name + '_distancevector')
            
            for jroot, jdirs, jfiles in os.walk(jsons_dir):
                json_files = jfiles
            assert len(json_files) == len(files), f'Number of videos {len(files)} is not equal to number of jsons {len(json_files)}'

            json_files, files, max_dist = check_integrity(jroot, root, json_files, files)
            max_dists.append(max_dist)
  
            video_dataset['videos'] += [os.path.join(root, file) for file in files]
            video_dataset['jsons'] += [os.path.join(jroot, file) for file in json_files]
            print(f'Dataset contains {len(video_dataset["videos"])} entries')

        return 'videos', video_dataset, np.max(max_dists)

    # for images
    if args.images:
        max_dist = 0
        images_dataset = dict({'images': [], 'labels': []})
        for root, dirs, files in os.walk(inputpath):
            try:
                name, ext = os.path.splitext(files[0])
            except:
                continue
            else:
                if ext not in ['.jpg', '.jpeg', '.bmp']:
                    #print('The files in folder are not images!')
                    continue

            files = natsort.natsorted(files)
            path_json = root.replace('coords_images', 'distancevector')
            separate_jsons = root.replace('coords_images', 'coords_distvecs')

            if not os.path.exists(separate_jsons):
                with open(path_json + '.json', "r") as read_file:
                    results = json.load(read_file)
                    max_dist = np.array([np.array(results).max(), max_dist]).max()
                if len(files) != len(results):
                    print(f'Warning: number of frames in the folder {root} is not compatible ', 
                    f'with the number of frames in json ({len(files)} / {len(results)}) ',
                    f'this pair will be excluded from dataset')
                    pass
                else:
                    print(f'Creating new dir {separate_jsons}')
                    os.makedirs(separate_jsons)
                    create_separete_jsosns(separate_jsons, results)
            else:
                max_dist = 2.37

            for jrt, jdirs, jfls in os.walk(separate_jsons):
                if jfls:
                    jsons = natsort.natsorted(jfls)

            for img_path, lbl_path in zip(files, jsons):
                images_dataset['images'].append(os.path.join(root, img_path))
                images_dataset['labels'].append(os.path.join(jrt, lbl_path))

        return 'images', images_dataset, max_dist

    # in case there is another option
    else:
        raise NotImplementedError



if __name__ == "__main__":
    #conv_model = ConvolutionalVisionTransformer(num_classes=32, in_chans=3, spec=config['MODEL']['SPEC']).to(args.device)   
    conv_model = Encoder(
            in_channels=3,
            out_channels=128,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            block_out_channels=(64, 64),
            layers_per_block=3,
            act_fn='silu',
            norm_num_groups=32,
            double_z=False).to(args.device)
    vae_model = AutoEncoder(136, hid_dim=128).to(args.device).apply(init_weights)

    #if args.load_pretrained:
    #    print('Loading model from {args.checkpoint}...')
    #    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    if len(args.gpus) > 1:
        conv_model = torch.nn.DataParallel(conv_model, device_ids=args.gpus)
        vae_model = torch.nn.DataParallel(vae_model, device_ids=args.gpus)
    else:
        conv_model.to(args.device)
        vae_model.to(args.device)

    optim_conv = torch.optim.Adam(conv_model.parameters(), lr=1e-4, weight_decay = 1e-8)
    optim_vae = torch.optim.Adam(vae_model.parameters(), lr=1e-3, weight_decay = 1e-8)

    if args.pretrained:
        print('Loading model from {args.checkpoint}...')
        loads_conv = torch.load(f'./{args.check_load}/conv/best.pth', map_location=args.device)
        loads_vae = torch.load(f'./{args.check_load}/vae/best.pth', map_location=args.device)
        conv_model.load_state_dict(loads_conv['model'])
        vae_model.load_state_dict(loads_vae['model'])
        optim_conv.load_state_dict(loads_conv['optim'])
        optim_vae.load_state_dict(loads_vae['optim']) 

    #optim_conv = torch.optim.Adam(conv_model.parameters(), lr=1e-4, weight_decay = 1e-8)
    #optim_vae = torch.optim.Adam(vae_model.parameters(), lr=1e-3, weight_decay = 1e-8)

    loss_fn_conv = torch.nn.HuberLoss()
    loss_fn_vae = torch.nn.HuberLoss()

    mode, dataset, max_dist = check_input(args.inputpath)
    datalen = len(dataset[mode])
    print(f'Dataset length is {datalen}')

    best_conv = 0.0
    best_vae = 0.0

    loader = get_loader(dataset, mode, args, max_dist)
    for e in range(args.epochs):
        vae_coss, conv_coss, conv_mses = [], [], []
        for it, (image, label) in enumerate(loader):
            image = image.to(args.device)
            label = label.to(args.device)

            hid_predict = conv_model(image)
            dist_predict, hid = vae_model(label)

            #loss_vae = loss_fn_vae(dist_predict, label)
            #loss_conv = loss_fn_conv(hid_predict, hid)
            loss = loss_fn_vae(dist_predict, label) + \
                   (1-args.sigma)*loss_fn_conv(hid_predict.detach(), hid) + \
                   args.sigma * loss_fn_conv(hid_predict, hid.detach())

            optim_vae.zero_grad()
            optim_conv.zero_grad()
            loss.backward()
            optim_conv.step()
            optim_vae.step()
            #writer.add_scalar('loss/convNet', loss.item(), e*datalen+it)

            #optim_vae.zero_grad()
            #loss_vae.backward()
            #optim_vae.step()
            #writer.add_scalar('loss/VAE', loss_vae.item(), e*datalen+it)

            #print(f'At epoch {e} iteration {it} the conv_loss: {loss_conv.item():.5f} vae_loss: {loss_vae.item():.5f}')
            vae_cos = F.cosine_similarity(dist_predict.detach(), label.detach()).abs().mean().cpu().numpy()
            vec_cos = F.cosine_similarity(hid_predict.detach(), hid.detach()).abs().mean().cpu().numpy()
            mse_conv = F.l1_loss(hid_predict.detach(), hid.detach()).mean().cpu().numpy()
            conv_coss.append(vec_cos)
            vae_coss.append(vae_cos)
            conv_mses.append(mse_conv)

            if not it % 100:
                ep_conv = np.mean(conv_coss)
                ep_vae = np.mean(vae_coss)
                ep_mse = np.mean(conv_mses)
                conv_coss, vae_coss, conv_mses = [], [], []

                writer.add_scalar('loss/convNet', loss.item(), e*datalen+it)
                writer.add_scalar('iteration/cos_convNet_predict_true', ep_conv, it)
                writer.add_scalar('iteration/cos_vae_predict_true', ep_vae, it)
                writer.add_scalar('iteration/mae_convNet_predict_true', ep_mse, it)

                torch.save({'model': conv_model.state_dict(), 'optim': optim_conv.state_dict()}, 
                               f'./{args.check}/conv/ordinary.pth')
                torch.save({'model': vae_model.state_dict(), 'optim': optim_vae.state_dict()}, 
                               f'./{args.check}/vae/ordinary.pth')
                if best_conv < ep_conv:
                    best_conv = ep_conv
                    torch.save({'model': conv_model.state_dict(), 'optim': optim_conv.state_dict()}, 
                               f'./{args.check}/conv/best.pth')
                if best_vae < ep_vae:
                    best_vae = ep_vae
                    torch.save({'model': vae_model.state_dict(), 'optim': optim_vae.state_dict()},
                               f'./{args.check}/vae/best.pth')

                print(f'Distance for dist_vector is {vec_cos}, for vae_vector is {vae_cos}')
                print(hid[0], hid_predict.detach()[0])

        #ep_conv = np.mean(conv_coss)
        #ep_vae = np.mean(vae_coss)
        #ep_mse = np.mean(conv_mses)

        #writer.add_scalar('epoch/cos_convNet_predict_true', ep_conv, e)
        #writer.add_scalar('epoch/cos_vae_predict_true', ep_vae, e)
        #writer.add_scalar('epoch/mae_convNet_predict_true', ep_mse, e)
        #print(f'Distance for dist_vector is {vec_cos}, for vae_vector is {vae_cos}')
        #print(hid[0], hid_predict.detach()[0])
