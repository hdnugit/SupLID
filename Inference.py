import argparse, os, csv, random
from collections import OrderedDict
import torch.optim
from tqdm import tqdm
from utils.pyt_utils import eval_ood_measure
import numpy
import torchvision
from src.dataset.dataset_graphs import dataset_graphs
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from skimage.segmentation import mark_boundaries

#import torch
from sklearn.cluster import KMeans
import faiss

import numpy as np
import faiss
from copy import deepcopy
from torch_geometric.loader import DataLoader
import shutil


def knn_score(feas_train, feas, k=10, min=False, device='cuda'):
    # Ensure tensors are on the correct device
    feas_train = feas_train.to(device)
    feas = feas.to(device)

    # Compute cosine similarity (equivalent to FAISS IndexFlatIP)
    # feas_train = torch.nn.functional.normalize(feas_train, dim=1)
    # feas = torch.nn.functional.normalize(feas, dim=1)
    similarity = torch.matmul(feas, feas_train.T)  # Compute similarity matrix

    # Get top-k similarity values
    D, _ = torch.topk(similarity, k, dim=1, largest=True, sorted=True)

    # Compute scores based on min or mean
    scores = D.min(dim=1).values if min else D.mean(dim=1)

    return scores

def mle_batch(data, batch, k, eps=1e-12):
    k = min(k, len(data) - 1)
    
    dists = torch.cdist(batch, data)
    dists, _ = torch.sort(dists, dim=1) 
    r = dists[:, 1:k+1] 
    r_k = r[:, -1].unsqueeze(1) 
    lid = -k / torch.sum(torch.log((r ) / (r_k )), dim=1)
    return lid  

    
def compute_anomaly_score(base_conf, baseconf_name, feas, energy_train, feas_train, lids_train, labels_train, mode):

    if mode == 'base_conf_pixel':
        anomaly_score = base_conf # pass for calculating later       
    elif mode == 'base_conf_sup':
        if baseconf_name=='energy':
            anomaly_score = base_conf
        elif baseconf_name=='entropy':
            anomaly_score = base_conf                
        elif baseconf_name=='msp':
            anomaly_score = base_conf
        elif baseconf_name=='maxlogit':
            anomaly_score = base_conf
        elif baseconf_name=='kl':
            anomaly_score = -base_conf          
        
    elif mode == 'knn':
        anomaly_score = -knn_score(feas_train,feas,400)         
    elif mode == 'nnguide': 
        confs_train = -1. * energy_train
        scaled_feas_train = feas_train * confs_train[:, None]
        confs = -1 * base_conf        
        guidances =  knn_score(scaled_feas_train, feas, k=400)
        anomaly_score = -1 * guidances*confs                           
    elif mode == 'supLID': 
        feas = feas.cuda()
        confs_train = 1. * lids_train     
        scaled_feas_train = feas_train * confs_train[:, None] 
        if baseconf_name=='energy':
            confs = -1 * base_conf       
            guidances = (mle_batch(scaled_feas_train,feas,400))         
            anomaly_score = -1 * guidances*confs 
        elif baseconf_name=='entropy':
            confs = base_conf       
            guidances = 1/(mle_batch(scaled_feas_train,feas,400))         
            anomaly_score =  guidances*confs                 
        elif baseconf_name=='msp':
            confs = -1 * base_conf       
            guidances = (mle_batch(scaled_feas_train,feas,400))         
            anomaly_score = -1 * guidances*confs 
        elif baseconf_name=='maxlogit':
            confs = -1 * base_conf       
            guidances = (mle_batch(scaled_feas_train,feas,400))         
            anomaly_score = -1 * guidances*confs 
        elif baseconf_name=='kl':
            confs = base_conf       
            guidances = (mle_batch(scaled_feas_train,feas,400))         
            anomaly_score =  -1 * guidances*confs            
   

    return anomaly_score


def valid_anomaly(test_set, data_name, energy_train, feas_train, lids_train, labels_train, measure_way, baseconf_name):

    print("validating {} dataset with {} ...".format(data_name, measure_way))

    tbar = tqdm(list(range(len(test_set))), ncols=137, leave=True, miniters=1) 
    ood_gts_list = []
    anomaly_score_list = []
    feas_test_list = []
    y_list = []


    with torch.no_grad():
        for idx in tbar: 
            data, fname = test_set[idx]
            print(fname)
            label = data.mask
            #print(label.shape)
            if baseconf_name=='energy':
                out =  data.energy
            elif baseconf_name=='entropy':
                out =  data.entropy                
            elif baseconf_name=='msp':
                out =  data.msp
            elif baseconf_name=='maxlogit':
                out =  data.maxlogit
            elif baseconf_name=='kl':
                out =  data.kl  
                
            y = data.y
            
            #print(out.shape)
            feas = F.normalize(data.x, dim=1)

            
            anomaly_score = compute_anomaly_score(out, baseconf_name, feas, energy_train, feas_train, lids_train, labels_train, mode=measure_way).cpu()

            anomaly_score = anomaly_score[data.superpixels]
            
            # uncomment this for pixel performance
            # if baseconf_name=='energy':
            #     anomaly_score = data.ori_energy.detach().cpu()
            # elif baseconf_name=='entropy':
            #     anomaly_score = data.ori_entropy.detach().cpu()                
            # elif baseconf_name=='msp':
            #     anomaly_score = data.ori_msp.detach().cpu()
            # elif baseconf_name=='maxlogit':
            #     anomaly_score = data.ori_maxlogit.detach().cpu()
            # elif baseconf_name=='kl':
            #     anomaly_score = -data.ori_kl.detach().cpu()              
            
            
            # regular gaussian smoothing
            anomaly_score = anomaly_score.unsqueeze(0)
            anomaly_score = torchvision.transforms.GaussianBlur(7, sigma=1)(anomaly_score)
            anomaly_score = anomaly_score.squeeze(0)
            
            assert ~torch.isnan(anomaly_score).any(), "expecting no nan in score {}.".format(measure_way)
    
            ood_gts_list.append(numpy.expand_dims(label, 0))
            anomaly_score_list.append(numpy.expand_dims(anomaly_score.numpy(), 0))
            feas_test_list.append(feas)
            y_list.append(y)
            torch.cuda.empty_cache()


    # evaluation
    ood_gts = numpy.array(ood_gts_list)
    anomaly_scores = numpy.array(anomaly_score_list)

    roc_auc, prc_auc, fpr = eval_ood_measure(anomaly_scores, ood_gts, train_id_in, train_id_out)

    print("AUROC score for {}: {:.4f}".format(data_name, roc_auc))
    print("AUPRC score for {}: {:.4f}".format(data_name, prc_auc))
    print("FPR@TPR95 for {}: {:.4f}".format(data_name, fpr))

    with open('beoe_'+measure_way+data_name+'.csv', 'a+') as fl1:
        writer = csv.writer(fl1)  
        writer.writerow([data_name, fpr * 100, prc_auc * 100, roc_auc * 100]) 


    feas_test = torch.cat(feas_test_list, dim=0)
    y = torch.cat(y_list, dim=0)
    feas_test_ood = feas_test[y == train_id_out]
    feas_test_id = feas_test[y == train_id_in]
    
    return feas_test_ood, feas_test_id


def create_coreset(train_set, batch_size=20, device="cuda"):
    energy_train_list = []
    entropy_train_list = []
    feas_train_list = []
    label_list = []

    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for idx, (data, fnm) in enumerate(dataloader):

            # Move data to GPU if available
            data.x = data.x.to(device)
            data.energy = data.energy.to(device)
            data.entropy = data.entropy.to(device)
            data.y = data.y.to(device)

            feas = F.normalize(data.x, dim=1) 

            energy_train_list.append(data.energy)
            entropy_train_list.append(data.entropy)
            feas_train_list.append(feas)
            label_list.append(data.y)
            # torch.cuda.empty_cache()
            # del data, feas
            
            # Stop early after n batches
            print(idx)
            if idx == 0:  #0
                break

    # Concatenating all collected data
    energy_train = torch.cat(energy_train_list, dim=0)
    entropy_train = torch.cat(entropy_train_list, dim=0)
    feas_train = torch.cat(feas_train_list, dim=0)
    labels = torch.cat(label_list, dim=0)

    # Finding top 100 highest energies and corresponding features for each class
    top_energy_per_class = []
    top_entropy_per_class = []
    top_feas_per_class = []
    top_labels_per_class = []
    top_lids_per_class = []
    top_knns_per_class = []

    for label in range(19):
        class_mask = labels == label
        class_indices = class_mask.nonzero(as_tuple=True)[0]

        if class_indices.numel() == 0:  # Skip if no samples for the class
            continue

        class_energy = energy_train[class_indices]
        class_entropy = entropy_train[class_indices]
        class_feas = feas_train[class_indices]
        class_labels = labels[class_indices]
        print(class_feas.shape)
        
        # Compute LID (assuming `mle_batch` works with CUDA tensors)
        results = []
        for batch_chunk in torch.split(class_feas, 256):  # Process in chunks of 256
            results.append(mle_batch(class_feas, batch_chunk, 400))
        class_lid = torch.cat(results, dim=0)

        # Get top 100 based on LID
        top_indices = torch.argsort(class_lid, descending=False)[:400] #100 #class_lid
        top_energy_per_class.append(class_energy[top_indices])
        top_entropy_per_class.append(class_entropy[top_indices])
        top_feas_per_class.append(class_feas[top_indices])
        top_labels_per_class.append(class_labels[top_indices])
        top_lids_per_class.append(class_lid[top_indices])
        # top_knns_per_class.append(class_knn[top_indices])

    # Concatenate results
    top_energy = torch.cat(top_energy_per_class, dim=0)
    top_entropy = torch.cat(top_entropy_per_class, dim=0)
    top_feas = torch.cat(top_feas_per_class, dim=0)
    top_labels = torch.cat(top_labels_per_class, dim=0)
    top_lids = torch.cat(top_lids_per_class, dim=0)
    print(top_energy.shape, top_feas.shape)    
    return top_energy, top_entropy, top_feas, top_labels, top_lids, top_lids



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for baseconf_name in ['kl', 'entropy', 'energy','msp', 'maxlogit']:

        for measure_way in ["supLID"]: 
    
        
            for rm in [1]:
                parser = argparse.ArgumentParser(description='Anomaly Segmentation')
                parser.add_argument('--random_seed_data', default=rm, type=int)
                args = parser.parse_args()
        
                # Reproducing same results
                # torch.use_deterministic_algbeoethms(True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.enabled = False
                random.seed(args.random_seed_data)
                os.environ['PYTHONHASHSEED'] = str(args.random_seed_data)
                np.random.seed(args.random_seed_data)
                torch.manual_seed(args.random_seed_data)
                torch.cuda.manual_seed(args.random_seed_data)
                torch.cuda.manual_seed_all(args.random_seed_data)
        
                
                train_set = dataset_graphs(root='/data/scratch/projects/punim1942/segmentation_datasets/gnn_segements_io/cityscapestrain_super_graphs_beoe/')
                energy_train, entropy_train, feas_train, labels_train, lids_train, knns_train = create_coreset(train_set=train_set)
            

                for dset_str in ["smiyc_anomalyval", "smiyc_obstacleval", "roadanomalyval", "fs_staticval", "fs_lnfval"]:   
                    datloader = dataset_graphs(root='/data/scratch/projects/punim1942/segmentation_datasets/gnn_segements_io/'+dset_str+'_super_graphs_beoe/')
                
                    feas_test_ood, feas_test_id = valid_anomaly(test_set=datloader, data_name=dset_str, energy_train=energy_train, feas_train=feas_train, lids_train= lids_train, labels_train=labels_train, measure_way=measure_way, baseconf_name=baseconf_name)
        


train_id_in = 0
train_id_out = 1

if __name__ == '__main__':
    main()




