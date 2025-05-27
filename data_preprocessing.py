import os, time
import torch
import torch_geometric, torchvision
import matplotlib.pyplot as plt
from skimage.segmentation import slic
import numpy as np
from scipy import ndimage, stats
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
import torch.nn.functional as F
import skimage.io
import pickle
from PIL import Image
import sys
import numpy
from src.model_utils_old import load_network
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected
from torch_geometric.utils import degree
from src.imageaugmentations import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from src.dataset.cityscapes import Cityscapes
from src.dataset.cityscapes_first6rrs import Cityscapes_first
from src.dataset.coco import COCO
from src.dataset.validation.fishyscapes import Fishyscapes
from src.dataset.validation.lost_and_found import LostAndFound
from src.dataset.validation.road_anomaly import RoadAnomaly
from src.dataset.validation.segment_me_if_you_can import SegmentMeIfYouCan
from model.network import Network
from collections import OrderedDict
from skimage.measure import regionprops
import torch_scatter
import gc
gc.collect()

def mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    m = counts.argmax()
    return values[m]
    
def get_mean(arr):
    m = torch.mean(arr,0)
    return m

def create_superpixels(output_file_path, input_dataset, model, mean, std):
    torch.cuda.empty_cache() 
    gc.collect()
    for ij in range(len(input_dataset)):   
        image, target, img_name = input_dataset[ij]      
        print(ij)
        start_time = time.time()
        
        #Hyperparameters
        num_superpixels = int(image.shape[1]*image.shape[2] / 200)
        
        mask = np.array(target)
        
        ori_img=image.clone()
        for i in range(ori_img.shape[0]):
            ori_img[i] = ori_img[i].mul(std[i]).add(mean[i])
        ori_img = np.array(ori_img.permute(1,2,0))

        
        superpixels = slic(ori_img.astype('double')/np.max(ori_img), n_segments = num_superpixels, compactness = 1, sigma = 3, min_size_factor = 0, start_label=0, enforce_connectivity=False, channel_axis=2)
        num_nodes = np.amax(superpixels)+1

        logits, ft = model(image.unsqueeze(0).cuda())
        ft = torch.squeeze(ft).permute(1, 2, 0)
        logits= logits.squeeze()[:19]       
        maxlogit = -1.0* logits.max(dim=0)[0]
        msp = -1.0* F.softmax(logits, dim=0).max(dim=0)[0]
        energy = -(1.0 * torch.logsumexp(logits, dim=0))
        #print('energy',energy.shape)
        prob = torch.softmax(logits, dim=0)
        # print(prob.shape)
        predicted_classes = torch.argmax(prob, dim=0)
        # print(predicted_classes.shape)
        entropy = -torch.sum(prob * torch.log(prob), dim=0) / torch.log(torch.tensor(19.))       
        
        log_probs = F.log_softmax(logits, dim=0)
        uniform = torch.full_like(log_probs, 1.0 / logits.shape[0])  # [C, H, W]
        ori_kl = -torch.sum(uniform * log_probs, dim=0)  # shape: [H, W]  

        ft_flat = ft.view(-1, ft.size(-1))  # shape: (720*1280, 304)
        superpixels_flat = torch.tensor(superpixels.flatten(), device=ft.device)
        x = torch_scatter.scatter_mean(ft_flat, superpixels_flat, dim=0)        
        
        e = ndimage.labeled_comprehension(energy.detach().cpu().numpy(), labels=superpixels, func=np.mean, index=range(0, num_nodes), out_dtype='float32', default=-1.0)
        e = torch.from_numpy(e).cuda()
        p = ndimage.labeled_comprehension(entropy.detach().cpu().numpy(), labels=superpixels, func=np.mean, index=range(0, num_nodes), out_dtype='float32', default=-1.0)
        p = torch.from_numpy(p).cuda()
        m = ndimage.labeled_comprehension(msp.detach().cpu().numpy(), labels=superpixels, func=np.mean, index=range(0, num_nodes), out_dtype='float32', default=-1.0)
        m = torch.from_numpy(m).cuda()
        ml = ndimage.labeled_comprehension(maxlogit.detach().cpu().numpy(), labels=superpixels, func=np.mean, index=range(0, num_nodes), out_dtype='float32', default=-1.0)
        ml = torch.from_numpy(ml).cuda()
        kl = ndimage.labeled_comprehension(ori_kl.detach().cpu().numpy(), labels=superpixels, func=np.mean, index=range(0, num_nodes), out_dtype='float32', default=-1.0)
        kl = torch.from_numpy(kl).cuda()

        predicted_classes = predicted_classes.cpu().detach().numpy()
        # print(mask.shape, predicted_classes.shape, predicted_classes.dtype)
        
        y = ndimage.labeled_comprehension(mask, labels=superpixels, func=mode, index=range(0,num_nodes), out_dtype='int32',default=-1.0)
        y = torch.from_numpy(y).cuda()
        
        pred = ndimage.labeled_comprehension(predicted_classes,labels=superpixels,func=mode,index=range(0,num_nodes),out_dtype='int32',default=-1.0)
        pred=torch.from_numpy(pred).cuda()
        
        data = Data(x=x, y=y, superpixels=superpixels, mask=mask, pred=pred, energy=e, entropy=p, msp=m, maxlogit=ml, kl=kl, ori_energy=energy, ori_entropy=entropy, ori_msp=msp, ori_maxlogit=maxlogit, ori_kl=ori_kl)
        
        print('time', time.time()-start_time)
        
        f = open(os.path.join(output_file_path,img_name), 'wb')
        pickle.dump(data,f )
        f.close()
        torch.cuda.empty_cache()
        del data
        
if __name__ == "__main__":
    import argparse
    for dst, spt in zip([ 'smiyc_anomaly','fs_static','smiyc_obstacle','roadanomaly', 'fs_lnf', 'cityscapes'],[ 'val', 'val', 'val', 'val', 'val', 'train']):
        parser = argparse.ArgumentParser(description='Generate graphs for input images.')   
        parser.add_argument('-dataset', '--dataset', type=str, default=dst ) #COCO+train, cityscapes+train/val, fs_lnf+val, fs_static+val, roadanomaly+val, smiyc_anomaly+val, smiyc_obstacle+val, laf+test
        parser.add_argument('-split', '--split', type=str, default=spt )   
        parser.add_argument('--ckpt_path', type=str, default='/data/scratch/projects/punim1942/segmentation_datasets/io/cityscapes/weights/')
        args = parser.parse_args()
    
        output_file_path = '/data/scratch/projects/punim1942/segmentation_datasets/gnn_segements_io/'+args.dataset+args.split+'_super_graphs_ori/'
        os.makedirs(output_file_path, exist_ok=True)

    

        #original ckpt
        net = load_network("DeepLabV3+_WideResNet38", 19, os.path.join(args.ckpt_path, "DeepLabV3+_WideResNet38.pth"))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        net.eval()
        
        """Normalization parameters"""
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = Compose([ToTensor(), Normalize(mean, std)])
    
        if args.dataset=='cityscapes':
            input_dataset = Cityscapes(root="/data/scratch/projects/punim1942/segmentation_datasets/cityscapes", split=args.split, transform=transform)
        elif args.dataset=='cityscapes_first':
            input_dataset = Cityscapes_first(root="/data/gpfs/projects/punim1942/energy_score_gnn_superpixel/images_first6randumrun", split=args.split, transform=transform)            
        elif args.dataset=='COCO':
            input_dataset = COCO(root="/data/scratch/projects/punim1942/segmentation_datasets/COCO/2017", split=args.split, transform=transform)  
        elif args.dataset=='laf':
            input_dataset = LostAndFound(root='/data/scratch/projects/punim1942/segmentation_datasets/lost_and_found', transform=transform)
        elif args.dataset=='fs_lnf':
            input_dataset = Fishyscapes(split='LostAndFound', root='/data/scratch/projects/punim1942/segmentation_datasets/fishyscapes', transform=transform)
        elif args.dataset=='fs_static':
            input_dataset = Fishyscapes(split='Static', root='/data/scratch/projects/punim1942/segmentation_datasets/fishyscapes', transform=transform)
        elif args.dataset=='roadanomaly':
            input_dataset = RoadAnomaly(root='/data/scratch/projects/punim1942/segmentation_datasets/road_anomaly', transform=transform) 
        elif args.dataset=='smiyc_anomaly':
            input_dataset = SegmentMeIfYouCan(split='road_anomaly', root='/data/scratch/projects/punim1942/segmentation_datasets/segment_me', transform=transform)
        elif args.dataset=='smiyc_obstacle':
            input_dataset = SegmentMeIfYouCan(split='road_obstacle', root='/data/scratch/projects/punim1942/segmentation_datasets/segment_me', transform=transform)    

        create_superpixels(output_file_path, input_dataset, net, mean, std)
        
      
