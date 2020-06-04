from torch.utils.data import Dataset
from PIL import Image


class Gallery:
    
    def __init__(self, **kwargs):
        self.lst = []
        self.not_seen_param = kwargs.pop("not_seen_param", 30)
        self.limit = kwargs.pop("limit", 1000)
        self.idx = 0

    def __len__(self):
        return len(self.lst)
    
    def append(self, features):
        """append id to list"""
        ReidEntity(self.idx, features)
        self.idx += 1
        
        # remove unactive elements if too many
        if len(self) > self.limit:
            self.purge()
    
    def yield_active(self):
        """generate actives query"""
        for x in self.lst:
            if x.active:
                yield x.features, x.id
    
    def get_list_actives(self):
        """get only active elements (feat+id)"""
        return [x for x in self.yield_active()]
    
    def len_act(self):
        """len of active elements list"""
        return sum([x.active for x in self.lst])

    def len(self):
        """how many unique elements"""
        return len(self.lst)
    
    def step(self):
        """unactivate old features"""
        for idt in self.lst:
            idt.not_seen_since += 1
            if idt.not_seen_since > self.not_seen_param:
                idt.active = False
    
    def purge(self):
        """clean list for memory"""
        for idt in self.lst:
            if not idt.active:
                self.lst.remove(idt)


class ReidEntity:
    """gallery identity for reid"""
    
    def __init__(self, id, features, center=(0, 0)):
        self.id = id  # identity number
        self.features = features  # 2048 features from backbone
        self.center = center  # position in the
        
        self.active = True
        self.not_seen_since = 0
    
    def update_features(self, new_features):
        self.features = new_features
        self.not_seen_since = 0
        self.active = True


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

