import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np

import net
import pickle
import torch.nn.functional as F
from torch.nn import init
import torch.utils.data as torch_data
from sampler import InfiniteSamplerWrapper
from function import adaptive_instance_normalization

def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if (content_f.size() != style_f.size()):
        print(content_f.size())
        print(style_f.size())
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 36, 5, stride=2, padding=1),
            nn.ELU(0.2, inplace=True),
            nn.Conv2d(36, 48, 3, stride=2, padding=1),
            nn.ELU(0.2, inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.ELU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ELU(0.2, inplace=True),
            nn.Dropout(p=0.5)           
        )
        self.output_layer = nn.Sequential(
            nn.Linear(3200,100),
            nn.ELU(0.2, inplace=True),
            nn.Linear(100,50),
            nn.ELU(0.2, inplace=True),
            nn.Linear(50,10),
            nn.ELU(0.2, inplace=True),
            nn.Linear(10,1),
            nn.Tanh()
        )
        self.reset_params()
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
            
    def forward(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        label = self.output_layer(feature_repr)
        return label

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class car_data(torch_data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        with open(root, 'rb') as fp:
            contents = pickle.load(fp)
            if train==True:
                self.data = contents['train']['imgs']
                self.label = contents['train']['steers']
            else:
                self.data = contents['test']['imgs']
                self.label = contents['test']['steers']
        
        self.shape = self.data.shape
        
    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return self.data.shape[0]

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='experiments_s2m/decoder_iter_30000.pth.tar')
# training options
parser.add_argument('--log_dir', default='./drive_logs_s2m',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=300000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--alpha', type=float, default=0.8,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
args = parser.parse_args()

device = torch.device('cuda')
writer = SummaryWriter(log_dir=args.log_dir)

decoder = net.decoder
vgg = net.vgg
classifier = Classifier()
classifier = classifier.cuda()

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
with open('cardata/sand.pkl', 'rb') as fp:
     source= pickle.load(fp)
with open('cardata/mountain.pkl', 'rb') as fp:
     target= pickle.load(fp)

T = transforms.Compose([transforms.Resize(size=(80)), transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
source_dset = car_data('cardata/mountain.pkl', transform=T)
T = transforms.Compose([transforms.Resize(size=(80)), transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
target_dset = car_data('cardata/sand.pkl', transform=T)

source_loader = torch_data.DataLoader(source_dset, batch_size=args.batch_size, shuffle=True)
target_loader = torch_data.DataLoader(target_dset, batch_size=args.batch_size, shuffle=True)
#index=1
#loss_ = []
#real_loss_ = []
for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    source_data, source_label = source_loader.__iter__().next()
    target_data, target_label = target_loader.__iter__().next()
    source_data = Variable(source_data).cuda()
    source_label = Variable(source_label).cuda()
    target_data = Variable(target_data).cuda()
    target_label = Variable(target_label).cuda()
    source_label = (source_label).type(torch.cuda.FloatTensor)
    target_label = (target_label).type(torch.cuda.FloatTensor)

        
    fake_img = style_transfer(vgg, decoder, source_data, target_data,
                                args.alpha)

    optimizer.zero_grad()
    label_pred_G = classifier(fake_img)
    label_pred_G = label_pred_G.view(args.batch_size)
    loss = F.mse_loss(label_pred_G, source_label)
    loss.backward()
    optimizer.step()
    label_pred_T = classifier(source_data)
    label_pred_T = label_pred_T.view(args.batch_size)
    real_loss = F.mse_loss(label_pred_T, target_label)
    #loss_.append(loss.item())
    #real_loss_.append(real_loss.item())
    writer.add_scalar('loss', loss.item(), i+1)
    writer.add_scalar('real_loss', real_loss.item(), i+1)
    #index = index+1
    if i%100 == 0:
        #print("train_loss : ", np.mean(loss_))
        #print("real_loss", np.mean(real_loss_))
        torch.save(classifier, './model_drive_s2m')
        #loss_ = []
        #real_loss_ = []

torch.save(classifier, './model_drive_s2m')
