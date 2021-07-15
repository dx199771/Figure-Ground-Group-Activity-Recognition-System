import os

from utils import video_transformer

from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim

from utils.video2seq import Video_to_seq as Video_to_seq
from utils.dataloader import Groundstream_dataloader, Figurestream_dataloader
from nets.I3D import InceptionI3d
from nets.figurestream_CNN import FigureCNN
import argparse
from utils.util import *
from train import training
# parameters
parser = argparse.ArgumentParser(description='Base Processor')

# processor
parser.add_argument('--data_json', type=str, default="data.json", help='training data information json file path')
parser.add_argument('--batch_size', type=int, default=2, help='number of batch size')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--device', type=int, default=0, help='processing device (GPU or not)')
parser.add_argument('--optim', type=str, default="Adam", help='optimizer type')
parser.add_argument('--base_lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay rate')

parser.add_argument('--dsize', type=int, default=640, help='image normalization size')
parser.add_argument('--num_classes', type=int, default=6, help='numbers of class in training samples')
parser.add_argument('--pretrained', type=str, default="./models/rgb_charades.pt", help='pre-trained model .pt file')

parser.add_argument('--resume', type=int, default=640, help='image normalization size')

opt = parser.parse_args()

path = "data.json"
if not os.path.exists("./data/videoData/seq3"):
    dataset = Video_to_seq(data=path,
                      root=r'F:\Disssertation\soccernet data\rawvideo',
                      output="seq3",
                      transforms=None)
train_transforms = transforms.Compose([
                                        video_transformer.RandomCrop(224),
                                       video_transformer.RandomHorizontalFlip(),
                                       ])
test_transforms = transforms.Compose([video_transformer.CenterCrop(224)])

train_transforms = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

ground_train_dataset = Groundstream_dataloader(folder_path="./data/VideoData/seq3",
                         txt_file_path="./data/VideoData/seq_data3.txt",
                         transforms=train_transforms,
                         num_classes=opt.num_classes,
                         )
figure_train_dataset = Figurestream_dataloader(folder_path="./data/SkeletonData/",
                         txt_file_path="./data/VideoData/seq_data3.txt",
                         num_classes=opt.num_classes,
                         )

#dataloader = torch.utils.data.DataLoader(ground_train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=False)
dataloader = torch.utils.data.DataLoader(figure_train_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=False)


ground_val_dataset = Groundstream_dataloader(folder_path="./data/VideoData/seq4",txt_file_path="./data/VideoData/seq_data4.txt",
                             transforms=train_transforms, num_classes=opt.num_classes)
figure_val_dataset = Groundstream_dataloader(folder_path="./data/VideoData/seq4",txt_file_path="./data/VideoData/seq_data4.txt",
                             transforms=train_transforms, num_classes=opt.num_classes)
val_dataloader = torch.utils.data.DataLoader(ground_val_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=False)
val_dataloader = torch.utils.data.DataLoader(figure_val_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=False)

dataloaders = {'train': dataloader, 'val': val_dataloader}
datasets = {'train': figure_train_dataset, 'val': figure_val_dataset}

# figure and ground models declaration
#ground_stream = InceptionI3d(num_classes=opt.num_classes, in_channels=3)

#ground_stream.load_state_dict(torch.load(opt.pretrained))
#ground_stream.replace_logits(6)
#ground_stream.cuda()
#ground_stream = nn.DataParallel(ground_stream)

figure_stream = FigureCNN(num_classes=opt.num_classes, use_pivot_distances=False)
figure_stream.cuda()



#figure_dataloader = {'train':, 'val':}
#figure_datasets = {'train': , 'val': }


# optimizer

if opt.optim == "SGD":
    optimizer = optim.SGD(
        params=figure_stream.parameters(),
        lr=opt.base_lr,
        momentum=0.9,
        weight_decay=opt.weight_decay
    )
elif opt.optim == "Adam":
    optimizer = optim.Adam(
        params=figure_stream.parameters(),
        lr=opt.base_lr,
        weight_decay=opt.weight_decay
    )

training(figure_stream, opt.num_epochs, dataloaders, optimizer, opt.num_classes)

#training(ground_stream, opt.num_epochs, dataloaders, optimizer, opt.num_classes)
"""



# STGCN model
Model = GCN()
#model =
loss = nn.CrossEntropyLoss()



#TODO adjust lr function
#TODO weights initialization












dataloader = I3d_dataloader(folder_path="./data/VideoData/seq",txt_file_path="./data/VideoData/seq_data.txt",transforms=train_transforms)

dataloader_ = torch.utils.data.DataLoader(dataloader, batch_size=batch_size, shuffle=False)


# preprocessing (convert video clip to pytorch .pt file and save)

i3d = I3D.I3D(num_classes=21)


for index, (inputs,labels) in enumerate(dataloader_):
    inputs = Variable(inputs.cuda())
    labels = Variable(labels.cuda())

    inputs = i3d(inputs)
    print(inputs.shape)

"""

#val_dataset = Dataset(train_split, root, mode, test_transforms)
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)