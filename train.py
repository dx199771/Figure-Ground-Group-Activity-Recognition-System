import torch

from torch.autograd import Variable
import torch.nn.functional as F
from utils.util import *
from tqdm import tqdm
import torch.nn as nn

def training(model, num_epochs, dataloaders, optimizer, num_classes ):
    for epoch in range(num_epochs):
        for phase in ['train','val']:
            print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            correct = 0
            criterion_groups = nn.CrossEntropyLoss(reduction='sum')

            """
            for index, (inputs, labels) in enumerate(tqdm(dataloaders[phase],
                                                total=len(dataloaders[phase].dataset)/2, desc ="Processing data:")):
                                                """
            for inputs,labels in dataloaders[phase]:
                # wrap data in variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                optimizer.zero_grad()
                output = model(inputs)
                print(output.shape,labels.long().shape)
                loc_loss = F.binary_cross_entropy_with_logits(output, labels)
                    #criterion_groups(output, labels.long())
                tot_loc_loss += loc_loss.item()
                #print(torch.max(output, dim=2)[0])
                #print(torch.max(torch.max(output, dim=2)[0],1)[1],torch.max(torch.max(labels, dim=2)[0],1)[1], correct)
                #print((torch.max(torch.max(output, dim=2)[0],1)[1]== torch.max(torch.max(labels, dim=2)[0],1)[1],correct))

                correct += sum(torch.max(output, dim=1)[1] == torch.max(labels, dim=1)[1])
                print(torch.max(output, dim=1)[1] , torch.max(labels, dim=1)[1])
                print(correct)
                loss = loc_loss

                tot_loss += loss.item()
                loss.backward()
                optimizer.step()
            print("{} {}".format(loc_loss, tot_loss/len(dataloaders[phase].dataset)))

                  #tot_loss/ len(dataloaders[phase].dataset))
            print("correct:",correct, " ",correct / len(dataloaders[phase].dataset))
                #tot_cls_loss += cls_loss.data[0]
"""

def training(model, num_epochs, dataloaders, optimizer, num_classes ):
    for epoch in range(num_epochs):
        for phase in ['train','val']:
            print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            correct = 0
            for index, (inputs, labels) in enumerate(tqdm(dataloaders[phase],
                                                total=len(dataloaders[phase].dataset)/2, desc ="Processing data:")):
                # wrap data in variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                labels.unsqueeze_(-1)

                labels = labels.expand(2,num_classes,50)
                inputs = inputs.permute(0,2,1,3,4)

                optimizer.zero_grad()
                t = inputs.size(2)
                output = model(inputs)
                #print(output[:,:3,:,0,0])

                output = F.interpolate(output, t, mode='linear')


                loc_loss = F.binary_cross_entropy_with_logits(output, labels)
                tot_loc_loss += loc_loss.item()

                cls_loss = F.binary_cross_entropy_with_logits(torch.max(output, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.item()
                #print(torch.max(output, dim=2)[0])
                #print(torch.max(torch.max(output, dim=2)[0],1)[1],torch.max(torch.max(labels, dim=2)[0],1)[1], correct)
                #print((torch.max(torch.max(output, dim=2)[0],1)[1]== torch.max(torch.max(labels, dim=2)[0],1)[1],correct))

                correct += sum(torch.max(torch.max(output, dim=2)[0],1)[1] == torch.max(torch.max(labels, dim=2)[0],1)[1])
                num_steps_per_update = 4
                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update

                tot_loss += loss.item()
                loss.backward()
                optimizer.step()
            print("{} {} {}".format(loc_loss, cls_loss, tot_loss/len(dataloaders[phase].dataset)))

                  #tot_loss/ len(dataloaders[phase].dataset))
            print("correct:",correct, " ",correct / len(dataloaders[phase].dataset))
                #tot_cls_loss += cls_loss.data[0]
                                                
"""