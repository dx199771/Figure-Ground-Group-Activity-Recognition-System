"""
    Author: Xu Dong
    Student Number: 200708160
    Email: x.dong@se20.qmul.ac.uk

    School of Electronic Engineering and Computer Science
    Queen Mary University of London, UK
    London, UK
"""

import argparse,random

# import torch libraries
import torch
import torch.optim as optim
from torchvision import transforms

from utils import video_transformer
from utils.video2seq import Video_to_seq as Video_to_seq
from utils.dataloader import Groundstream_dataloader, Figurestream_dataloader, Twostream_dataloader
from nets.groundstream_I3D import GroundI3D
from nets.groundstream_C3D import GroundC3D
from nets.figurestream_CNN import FigureCNN
from train import figure_training, ground_training, two_stream_training
from utils.util import *

# parameters
parser = argparse.ArgumentParser(description='Base Processor')

# processor
parser.add_argument('--data_json', type=str, default="./data/data.json", help='training data information json file path')
parser.add_argument('--batch_size', type=int, default=100, help='number of batch size')
parser.add_argument('--ground_batch_size', type=int, default=2, help='number of batch size of ground stream')

parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--device', type=int, default=0, help='processing device (GPU or not)')
parser.add_argument('--optim', type=str, default="Adam", help='optimizer type')
parser.add_argument('--base_lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay rate')

parser.add_argument('--dsize', type=int, default=640, help='image normalization size')
parser.add_argument('--num_classes', type=int, default=6, help='numbers of class in training samples')
parser.add_argument('--pretrained', type=str, default="./models/best_figure.pt", help='pre-trained model .pt file')
parser.add_argument('--variant', type=int, default=1, help='1:GroundI3D,2:GroundC3D,3:FigureCNN,4:TwostreamI3D,4,TwostreamC3D')
opt = parser.parse_args()

# seed setup
torch.manual_seed(5)
torch.cuda.manual_seed(5)
np.random.seed(5)
random.seed(5)


if not os.path.exists("./data/videoData/seq4"):
    dataset = Video_to_seq(data=opt.path,
                      root=r'F:\Disssertation\soccernet data\rawvideo',
                      output="seq3",
                      transforms=None)

# figure stream data pre-processing
figure_transforms = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
# train stream data pre-processing
train_transforms = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

# figure stream train dataset
figure_train_dataset = Figurestream_dataloader(folder_path="./data/SkeletonData/seq3",
                         txt_file_path="./data/VideoData/seq_data3.txt",
                         num_classes=opt.num_classes,
                         )
# figure stream validation dataset
figure_val_dataset = Figurestream_dataloader(folder_path="./data/SkeletonData/seq4",
                         txt_file_path="./data/VideoData/seq_data4.txt",
                         num_classes=opt.num_classes,
                         )

# ground stream train dataset
ground_train_dataset = Groundstream_dataloader(folder_path="./data/VideoData/seq3",
                         txt_file_path="./data/VideoData/seq_data3.txt",
                         transforms=train_transforms,
                         num_classes=opt.num_classes,
                                              )

# ground stream validation dataset
ground_val_dataset = Groundstream_dataloader(folder_path="./data/VideoData/seq4",
                         txt_file_path="./data/VideoData/seq_data4.txt",
                         transforms=train_transforms,
                         num_classes=opt.num_classes,
                                             )

# twostream stream train dataset
twostream_train_dataset = Twostream_dataloader(img_folder_path="./data/VideoData/seq3",
                         skeleton_folder_path="./data/SkeletonData/seq3",
                         txt_file_path="./data/VideoData/seq_data3.txt",
                         transforms=train_transforms,
                         num_classes=opt.num_classes,
                                              )

# twostream stream validation dataset
twostream_val_dataset = Twostream_dataloader(img_folder_path="./data/VideoData/seq4",
                         skeleton_folder_path="./data/SkeletonData/seq4",
                         txt_file_path="./data/VideoData/seq_data4.txt",
                         transforms=train_transforms,
                         num_classes=opt.num_classes,
                                              )

# figure train dataloader
figure_train_dataloader = torch.utils.data.DataLoader(figure_train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=False)
# figure val dataloader
figure_val_dataloader = torch.utils.data.DataLoader(figure_val_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=False)
# ground stream train dataloader
ground_train_dataloader = torch.utils.data.DataLoader(ground_train_dataset, batch_size=opt.ground_batch_size, shuffle=False, pin_memory=False)
# ground stream val dataloader
ground_val_dataloader = torch.utils.data.DataLoader(ground_val_dataset, batch_size=opt.ground_batch_size, shuffle=False, pin_memory=False)
# twostream stream train dataloader
twostream_train_dataloader = torch.utils.data.DataLoader(twostream_train_dataset, batch_size=opt.ground_batch_size, shuffle=True, pin_memory=False)
# twostream stream val dataloader
twostream_val_dataloader = torch.utils.data.DataLoader(twostream_val_dataset, batch_size=opt.ground_batch_size, shuffle=True, pin_memory=False)


# figure stream network
stream = FigureCNN(num_classes=opt.num_classes, use_pivot_distances=False,two_stream=False)
# I3Dground stream network
stream = GroundI3D(two_stream=False, num_classes=opt.num_classes, in_channels=3)
# C3D ground stream network
stream = GroundC3D(num_classes=opt.num_classes)

# load pre-trained model
stream.load_state_dict(torch.load(opt.pretrained))

# optimizer
if opt.optim == "SGD":
    optimizer = optim.SGD(
        params=stream.parameters(),
        lr=opt.base_lr,
        momentum=0.9,
        weight_decay=opt.weight_decay
    )
elif opt.optim == "Adam":
    optimizer = optim.Adam(
        params=stream.parameters(),
        lr=opt.base_lr,
        weight_decay=opt.weight_decay
    )
# dataloader
figure_dataloaders = {'train': figure_train_dataloader, 'val': figure_val_dataloader}
ground_dataloaders = {'train': ground_train_dataloader, 'val': ground_val_dataloader}
twostream_dataloaders = {'train': twostream_train_dataloader, 'val': twostream_val_dataloader}

# LR scheduler (comment out for using)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.1)
#scheduler.step()

if opt.variant == 1:
    stream = GroundI3D(two_stream=False, num_classes=opt.num_classes, in_channels=3)
    ground_training(stream, opt.num_epochs, ground_dataloaders, optimizer, opt.num_classes)
elif opt.variant == 2:
    stream = GroundC3D(num_classes=opt.num_classes)
    ground_training(stream, opt.num_epochs, ground_dataloaders, optimizer, opt.num_classes,opt.batch_size)
elif opt.variant == 3:
    stream = FigureCNN(num_classes=opt.num_classes, use_pivot_distances=False, two_stream=False)
    figure_training(stream, opt.num_epochs, figure_dataloaders, optimizer, opt.num_classes, opt.batch_size)
elif opt.variant == 4:
    two_stream_training("./models/best_figure.pt", "./models/best_i3d.pt", opt.num_epochs, twostream_dataloaders, optimizer,
                        opt.num_classes,opt.batch_size)
else:
    two_stream_training("I3D","./models/best_figure.pt", "./models/best_c3d.pt", opt.num_epochs, twostream_dataloaders, optimizer,
                        opt.num_classes, opt.batch_size)











