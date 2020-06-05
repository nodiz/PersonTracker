import torch
import os

from dr_utils import clean_folder


class Gallery:
    
    def __init__(self, save_path=None, **kwargs):
        self.lst = []
        self.not_seen_param = kwargs.pop("not_seen_param", 50)
        self.limit = kwargs.pop("limit", 50)
        self.idx = 0
        
        self.save_path = save_path
    
    def __len__(self):
        """how many unique elements"""
        return len(self.lst)
    
    def find_item(self, item):
        """could use get_item but might end up breaking something"""
        for x in self.lst:
            if x.idv == item:
                return x
    
    @property
    def len_act(self):
        """len of active elements list"""
        return sum([x.active for x in self.lst])
    
    def append(self, features, img=None):
        """append id to list"""
        self.lst.append(ReidEntity(self.idx, features))
        
        if img is not None:
            clean_folder(os.path.join(self.save_path, f"id-{self.idx}/"))
            img.save(os.path.join(self.save_path, f"id-{self.idx}/") + f"{self.lst[-1].cnt}.png")
            self.lst[-1].cnt += 1
        
        self.idx += 1
        
        # delete unactive elements if too many
        if len(self) > self.limit:
            self.purge()
            
        return self.idx - 1
    
    def update(self, idx, features, img=None):
        """ update id features """
        item = self.find_item(idx)
        
        item.update_features(features)  # will also reset counters
        
        if img is not None:
            img.save(os.path.join(self.save_path, f"id-{idx}/") + f"{item.cnt}.png")
            item.cnt += 1
    
    def yield_active(self):
        """generate actives query"""
        for x in self.lst:
            if x.active:
                yield x
    
    def get_gallery(self):
        """get active elements (feat+id)"""
        gf, ids = [], []
        for person in self.yield_active():
            gf.append(person.features)
            ids.append(person.idv)
        return torch.stack(gf), ids
    
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
    
    def __init__(self, idv, features, center=(0, 0)):
        self.idv = idv  # identity number
        self.features = features  # 2048 features from backbone
        self.center = center  # position in the
        
        self.cnt = 0
        self.active = True
        self.not_seen_since = 0
    
    def update_features(self, new_features):
        self.features = new_features
        self.not_seen_since = 0
        self.active = True
