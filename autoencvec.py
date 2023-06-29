
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.linear_e1 = nn.Linear(in_dim, hid_dim)
        self.linear_e2 = nn.Linear(hid_dim, hid_dim // 2)
        self.linear_e3 = nn.Linear(hid_dim // 2, hid_dim // 4)
        self.linear_e4 = nn.Linear(hid_dim // 4, hid_dim // 8)
        self.linear_e5 = nn.Linear(hid_dim // 8, hid_dim // 16)
        self.do_e1 = nn.Dropout(0.2)
        self.do_e2 = nn.Dropout(0.2)
        self.do_e3 = nn.Dropout(0.2)   

        self.linear_d1 = nn.Linear(hid_dim // 16, hid_dim // 8)
        self.linear_d2 = nn.Linear(hid_dim // 8, hid_dim // 4)
        self.linear_d3 = nn.Linear(hid_dim // 4, hid_dim // 2)
        self.linear_d4 = nn.Linear(hid_dim // 2, hid_dim)
        self.linear_d5 = nn.Linear(hid_dim, in_dim)
        self.probs = nn.Sigmoid()
        self.do_d1 = nn.Dropout(0.2)
        self.do_d2 = nn.Dropout(0.2)
        self.do_d3 = nn.Dropout(0.2) 

    def fwd_enc(self, x):
        x = F.selu(self.linear_e1(x)) 
        x = F.selu(self.linear_e2(x))
        #x = self.do_e1(x)
        #x = F.selu(self.linear_e3(x))
        #x = self.do_e2(x)
        #x = F.selu(self.linear_e4(x))
        #x = self.do_e3(x)
        x = self.linear_e3(x)
        return x
    
    def fwd_dec(self, x):
        #x = F.selu(self.linear_d1(x))
        #x = self.do_d1(x)
        #x = F.selu(self.linear_d2(x))
        #x = self.do_d2(x)
        x = F.selu(self.linear_d3(x))
        #x = self.do_d3(x)
        x = F.selu(self.linear_d4(x))
        x = self.linear_d5(x)
        return self.probs(x) 

    def forward(self, x):
        c = self.fwd_enc(x)
        x = self.fwd_dec(c)
        return x, c


class Dataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item


class Datas:
    def __init__(self, path):
        self.data = self.load_data(path)
        self.data = self._norm()
        self.div = int(len(self.data) * 0.9)

    def _norm(self):
        max = torch.max(self.data)
        print(f'Maximum distance in data is {max}')
        self.data /= max
        return self.data

    def get_datasets(self):
        train = Dataset(self.data[:self.div])
        test = Dataset(self.data[self.div:])
        return train, test

    def load_data(self, input_path):
        for root, dirs, files in os.walk(input_path):
            try: 
                name = root.split('/')[-1].split('\\')[-1].split('_')[-1]
            except:
                continue
            else:
                if name == 'distancevector': 
                    pass
                else:
                    continue
            result = []
            for jsn in files: 
                json_path = os.path.join(root, jsn)

                with open(json_path, "r") as read_file:
                    sequence = json.load(read_file)
                    result.append(torch.FloatTensor(sequence))
            return torch.cat(result)


def collate(input):
    return torch.stack(input)


def train(epochs=100, batch_sz=32):
    data_getter = Datas('/capital/datasets/trainProgettoX/date01_color_distancevector')
    trainset, testset = data_getter.get_datasets()
    print(f'Train_dataset size is {len(trainset)}, test_dataset size is {len(testset)}')
    trainloader = torch.utils.data.DataLoader(dataset = trainset,
                                     batch_size = batch_sz, collate_fn=collate,
                                     shuffle = True, drop_last=True)

    testloader = torch.utils.data.DataLoader(dataset = testset,
                                     batch_size = batch_sz, collate_fn=collate,
                                     shuffle = False, drop_last=True) 

    model = AutoEncoder(136, hid_dim=128).to(device).apply(init_weights)
    loss_fn = torch.nn.HuberLoss()
    #loss_f2 = torch.nn.BCELoss()
    #loss_fn = torch.nn.L1Loss()
    #loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(),
                                lr = 1e-3,
                                weight_decay = 1e-8
                                )

    const_sim = F.cosine_similarity(testset[700], testset[701], dim=0).abs().mean()
    const_sim1 = F.cosine_similarity(testset[700], testset[1000], dim=0).abs().mean()

    for ep in range(epochs):
        losses = []
        for it, item in tqdm(enumerate(trainloader)):
            item = item.to(device)
            prediction = model(item)[0]
            loss = loss_fn(prediction, item) * 136
            #loss = loss_f2(prediction, item)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if not it % 500:
                l = loss.detach().cpu().numpy()
                losses.append(l)
                #print(f'Loss in iteration {it} is {l:.5f}')
        #print(f'After {ep} epoches mean loss is {np.mean(losses):.5f}, prediction error is {torch.abs(prediction.detach() - item.detach()).cpu().numpy().mean():.5f}')
        #print(f'results: {item[0].cpu()} vs {prediction[0].detach().cpu()}')
        pred_error = torch.abs(prediction.detach() - item.detach()).cpu().numpy().mean()
        cos_sim = F.cosine_similarity(prediction.detach(), item.detach()).abs().mean().cpu().numpy()

        test_errors = []
        cossims = []
        for teit, teitem in tqdm(enumerate(testloader)):
            teitem = teitem.to(device)
            with torch.no_grad():
                prediction = model(teitem)[0]

            error = torch.abs(prediction - teitem).cpu().numpy().mean()
            cos_test = F.cosine_similarity(prediction, teitem).abs().mean().cpu().numpy()
            test_errors.append(error)
            cossims.append(cos_test)
        #print(f'Test error is {np.mean(test_errors):.5f}')
        print(f'At {ep} epoche m_loss is {np.mean(losses):.5f}, Test error is {np.mean(test_errors):.5f} Seq2_err = {const_sim:.5f}, Seq300_err = {const_sim1:.5f} cosine_sim {cos_sim:.5f}, test_sim {np.mean(cossims):.5f}')

if __name__ == '__main__':
    train()
