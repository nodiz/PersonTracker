import os, sys, time
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

from reidLib.modeling.baseline import Baseline


class Reid:
    def __init__(self,
                 baseline_path='reidLib/',
                 backbone="WeightsReid/resnet50-19c8e357.pth",
                 num_classes=1041,
                 query_path=None):
        """init the reid"""
        self.baseline_path = baseline_path
        sys.path.append(baseline_path)
        sys.path.append(os.path.join(baseline_path,'modeling'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Baseline(num_classes, 1, backbone, 'bnneck', 'after', 'resnet50', 'imageNet')
        self.model.eval()
        self.model.to(self.device)
        
        self.load_param()  # load parameters by default
        self.load_transform()
        
        self.gallery = []
        self.gallery_length = 0
        self.query = []
        self.query_length = 0
        
        self.query_path = query_path

    def load_param(self,
                   trained_path='WeightsReid/WeightFab/center/resnet50_model_100.pth'):
        """load the parameters"""

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

        imdatas = ReidDataset(imgs_query, transform)

        def val_collate_fn(batch):
            return torch.stack(batch, dim=0)
        
        demo_loader = DataLoader(
            imdatas, batch_size=self.query_length, shuffle=False, num_workers=2,
            collate_fn=val_collate_fn
        )

        for batch in demo_loader:
            return batch.to(self.device)
        
    def get_gallery(self):
        """ get query images"""
        return [im['features'] for im in self.gallery], [im['id'] for im in self.gallery]
        
    def evaluate_query(self, query_path=None):
        """evluate query with in-cache gallery"""
        with torch.no_grad():
            if not self.query_path:
                query_path = self.query_path
                batch = self.get_query()
                
                # Calculating feature for query
                qf = self.model(batch)  # (bs, 2048)

                qf = torch.nn.functional.normalize(qf, dim=1, p=2)
                
                m, n, gf, gf_id = self.query_length, self.gallery_length, self.get_gallery()
                
                distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                          torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            
                distmat.addmm_(1, -2, qf, gf.t())
                distmat = distmat.cpu().numpy()
                
                """now we have distmat, TODO:
                - match label
                - update gallery
                - return matched labels
                """
                
                raise NotImplementedError
        

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
        
        
