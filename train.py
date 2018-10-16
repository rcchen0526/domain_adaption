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

import net
from sampler import InfiniteSamplerWrapper
from function import adaptive_instance_normalization
from torchvision.utils import save_image

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def train_transform():
    transform_list = [
        transforms.Resize(size=(80)),
        #transforms.RandomCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')

args = parser.parse_args()

device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

decoder = net.decoder
vgg = net.vgg

discriminator = net.Discriminator()
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
discriminator.cuda()

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    loss_c, loss_s = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s

    fake_img = style_transfer(vgg, network.decoder, content_images, style_images,
                                    args.alpha)
    loss0, loss1, loss3, loss5, loss6= discriminator(fake_img.detach())
    LD = ( torch.mean((loss0 - 1.)**2) +torch.mean((loss1 - 1.)**2) + \
            torch.mean((loss3 - 1.)**2) +torch.mean((loss5 - 1.)**2) +torch.mean((loss6 - 1.)**2) ) / 5.

    loss = loss_c + loss_s + LD

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    r_loss0, r_loss1, r_loss3, r_loss5, r_loss6= discriminator(style_images)
    f0_loss0, f0_loss1, f0_loss3, f0_loss5, f0_loss6= discriminator(fake_img.detach())
    f1_loss0, f1_loss1, f1_loss3, f1_loss5, f1_loss6= discriminator(content_images)
    optimizer_D.zero_grad()
    d_loss = ( torch.mean((r_loss0 - 1.)**2) +torch.mean((r_loss1 - 1.)**2) + \
    torch.mean((r_loss3 - 1.)**2) +torch.mean((r_loss5 - 1.)**2) +torch.mean((r_loss6 - 1.)**2) + \
    torch.mean(f0_loss0**2) +torch.mean(f0_loss1**2) +torch.mean(f0_loss3**2) +torch.mean(f0_loss5**2) +torch.mean(f0_loss6**2) + \
    torch.mean(f1_loss0**2) +torch.mean(f1_loss1**2) +torch.mean(f1_loss3**2) +torch.mean(f1_loss5**2) +torch.mean(f1_loss6**2) ) / 5.
    
    d_loss.backward()
    optimizer_D.step()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_scalar('LD', LD.item(), i + 1)
    writer.add_scalar('D_loss', d_loss.item(), i + 1)
    if i%100 == 0:
        save_image(content_images.detach(), 'data/source_%d.png' % (((i+1)/100)%10), nrow=2, normalize=True)
        save_image(fake_img.detach(), 'data/images_%d.png' % (((i+1)/100)%10), nrow=2, normalize=True)
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth.tar'.format(args.save_dir,
                                                           i + 1))
writer.close()
