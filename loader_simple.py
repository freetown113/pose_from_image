import torch
import cv2
import random
import json
import os
from torchvision import transforms
import numpy as np

class DatasetVideos:
    def __init__(self, data, item_type='mix', mb_size=4, transforms=None, target_transform=None):
        self.data = data
        self.desc = dict()
        self.type = item_type
        self.minibatch = mb_size
        self.transforms = transforms
        self.t_transf = target_transform
        self.length = len(data['videos'])

    def get_desc(self, name):
        try:
            self.desc[name]
        except:
            self.desc[name] = self.make_desc(name)
            return self.desc[name]
        else:
            return self.desc[name]

    def make_desc(self, name):
        desc = dict({name: {}})
        stream = cv2.VideoCapture(name)
        assert stream.isOpened(), 'Cannot capture source'
        desc[name]['datalen'] = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        desc[name]['fourcc'] = int(stream.get(cv2.CAP_PROP_FOURCC))
        desc[name]['fps'] = stream.get(cv2.CAP_PROP_FPS)
        desc[name]['frameSize'] = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        stream.release()
        return desc[name]
    
    def get_frame(self, index):
        # output = dict()
        output = []
        cntr = 0
        jsons = self.get_json_data()
        stream = cv2.VideoCapture(self.video_file)
        assert stream.isOpened(), 'Cannot capture source'

        while True:
            (grabbed, frame) = stream.read()

            if grabbed:
                if cntr in index:
                    target = np.array(jsons[cntr])
                    if self.transforms is not None:
                        frame = self.transforms(frame)
                    if self.t_transf is not None:
                        target = self.t_transf(target)
                    # output[self.video_file + '/' + str(cntr).zfill(5) + ".jpg"] = (frame, target)
                    output.append((frame, target))
                cntr += 1

            else:
                return output
        
    def build_json_path(self):
        path = self.video_file.replace('videos', 'distancevector')
        path = os.path.splitext(path)[0]
        return path + '.json'

    def get_json_data(self):
        json_path = self.build_json_path()
        output = []
        with open(json_path, "r") as read_file:
            results = json.load(read_file)
        #print(f'there are {len(results)} entries in {json_path}')
        return results

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        name = self.data['videos'][index]
        desc = self.get_desc(name)
        if desc['datalen'] < self.minibatch:
            name = self.data['videos'][(index + random.randint(self.length))%self.length]
            desc = self.get_desc(name)
        self.video_file = name
        if self.type == 'mix':
            idxs = random.sample(range(0, desc['datalen']), self.minibatch)
            items = self.get_frame(idxs)

        elif self.type == 'seq':
            index = random.randint(0, desc['datalen'] - self.minibatch)
            index = list(range(index, index + self.minibatch))
            items = self.get_frame(index)

        else:
            raise TypeError('Unknown type of item type provided: {self.type}')
        return items
    

def collateVid(input):
    images, labels = [], []
    for item in input:
        for (img, lbl) in item: 
            images.append(img)
            labels.append(lbl)
    return torch.stack(images), torch.stack(labels)



class DatasetImages:
    def __init__(self, data, transforms=None, target_transform=None):
        self.data = data
        self.transforms = transforms
        self.t_transf = target_transform

    def __len__(self):
        return len(self.data['images']) 
    
    def __getitem__(self, idx):
        lbl_item = self.data['labels'][idx]
        with open(lbl_item, "r") as read_file:
            label = json.load(read_file)
        if self.t_transf is not None:
            label = self.t_transf(label)

        img_item = self.data['images'][idx]
        image = cv2.imread(img_item)
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label      
    

def collateImg(input):
    images, labels = [], []
    for item in input:
        images.append(item[0])
        labels.append(item[1])
    return torch.stack(images), torch.stack(labels)


def get_loader(data, mode, config, max_dist):
    print(f'Max distance is {max_dist}')
    transf = transforms.Compose([transforms.ToTensor(),
                                #transforms.Resize(224) 
                                ])
    t_transf = transforms.Compose([lambda x: torch.FloatTensor(x),
                                   lambda x: x / max_dist
                                  ])

    if mode == 'videos':
        dataset = DatasetVideos(data, item_type=config.item_type, mb_size=config.mb, transforms=transf, target_transform=t_transf)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=config.dev_batch, 
                                                collate_fn=collateVid,
                                                shuffle=True, drop_last=True)
    elif mode == 'images':
        dataset = DatasetImages(data, transforms=transf, target_transform=t_transf)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=config.dev_batch * config.mb, 
                                                collate_fn=collateImg,
                                                shuffle=True, drop_last=True)
    
    return dataloader
