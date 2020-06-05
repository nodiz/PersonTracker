import os, sys, time
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time

from reidLib.modeling.baseline import Baseline

from reid_utils import *


class Reid:
    def __init__(self,
                 baseline_path='reidLib/',
                 backbone="weights/reid//resnet50-19c8e357.pth",
                 num_classes=1041,
                 threshold=0.5,
                 save_path=None,
                 verbose=False,
                 **kwargs):
        """init the reid"""
        
        self.baseline_path = baseline_path
        sys.path.append(baseline_path)
        sys.path.append(os.path.join(baseline_path, 'modeling'))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Baseline(num_classes, 1, backbone, 'bnneck', 'after', 'resnet50', 'imageNet')
        self.model.eval()
        self.model.to(self.device)
        
        self.load_param()
        self.load_transform()
        
        self.gallery = Gallery(save_path)  # if savepath!=none we will create folder for identities (nice++)
        self.save_path = save_path
        self.query = []
        
        self.threshold = threshold  # threshold for new identity
        
        self.verbose = verbose
    
    def load_param(self,
                   trained_path='weights/reid/resnet50_model_100.pth'):
        """load the parameters"""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            param_dict = torch.load(trained_path, map_location=self.device).state_dict()
            for i in param_dict:
                if 'classifier' in i:
                    continue
                self.model.state_dict()[i].copy_(param_dict[i])
    
    def load_transform(self):
        """transform for loading data"""
        normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform = T.Compose([
            T.Resize([256, 128]),
            T.ToTensor(),
            normalize_transform
        ])
    
    def get_query(self, query_path):
        """ get query images
        as it is right now, we would have just one batch assuming few detections in a frame.
        In case of very crowded environments this code could be easily expanded to eval on mini-batches
        """
        
        imgs_query = sorted(glob.glob(os.path.join(query_path, "*.png")))
        self.query_length = np.shape(imgs_query)[0]
        
        imdatas = ReidDataset(imgs_query, self.transform)
        
        def val_collate_fn(batch):
            return torch.stack(batch, dim=0)
        
        demo_loader = DataLoader(
            imdatas, batch_size=self.query_length, shuffle=False, num_workers=2,
            collate_fn=val_collate_fn
        )
        
        for batch in demo_loader:
            return batch.to(self.device)
    
    def get_query_from_list(self, list):
        t_list = []
        for img in list:
            if type(img) == np.ndarray:
                img = Image.fromarray(img)
            img = self.transform(img)
            t_list.append(img)
        return torch.stack(t_list, dim=0).to(self.device)
        
    def evaluate_query(self, query):
        """evluate query with in-cache gallery"""
        with torch.no_grad():
            if query == []:  # no reid needed
                return []
            
            if self.save_path is not None:
                query_pics = [Image.fromarray(img) for img in query]
    
            query = self.get_query_from_list(query)
                
            # Calculating feature for query
            start_time = time.time()
            
            qf = self.model(query)  # (bs, 2048)

            qf = torch.nn.functional.normalize(qf, dim=1, p=2)  # TODO probably dangerous to normalize since we have few elemtns, better to do it together with the gallery but more comexp

            if self.verbose:
                print(f"--- ReId: %s seconds for {qf.shape[0]}-{query.shape[0]} pics---" % (time.time() - start_time))

            lq = len(query)
            del query  # freeing some more space, big matrix is coming

            
            if len(self.gallery) == 0:  # if first passage
                for i, x in enumerate(qf):
                    self.gallery.append(x) if self.save_path is None else self.gallery.append(x, query_pics[i])
                return range(lq)
            
            start_time = time.time()
            m, n, (gf, gf_id) = qf.shape[0], self.gallery.len_act, self.gallery.get_gallery()
            
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.cpu().numpy()

            # distmat: rows: querys, collums: gallery
            # now we start the game
            query_idx = [float('nan') for x in range(m)]
            distmat = np.where(distmat < self.threshold, distmat, np.inf)

            for ass in range(min(m,n)):
                idx = np.unravel_index(np.argmin(distmat, axis=None), distmat.shape)
                if distmat[idx] == np.inf:
                    break  # no more association possible
                else:  # make association
                    query_idx[idx[0]] = gf_id[idx[1]]
                    self.gallery.update(gf_id[idx[1]], qf[idx[0]]) if self.save_path is None else self.gallery.update(gf_id[idx[1]], qf[idx[0]], query_pics[idx[0]])

                    # deactivate element
                    distmat[idx[0], :] = np.inf
                    distmat[:, idx[1]] = np.inf
    
            #  finally check the not assigned and create new identities
            for i, x in enumerate(qf):
                if np.isnan(query_idx[i]):
                    nid = self.gallery.append(x) if self.save_path is None else self.gallery.append(x, query_pics[i])
                    query_idx[i] = nid

            #  desactivate old values
            self.gallery.step()
            
            if self.verbose:
                print(f"--- Ass: %s seconds for g{n}-q{m} pics---" % (time.time() - start_time))
            
            return query_idx


class ReidDataset(Dataset):
    """Image Person ReID Dataset"""
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img