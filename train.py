"""
    Author: Xu Dong
    Student Number: 200708160
    Email: x.dong@se20.qmul.ac.uk

    School of Electronic Engineering and Computer Science
    Queen Mary University of London, UK
    London, UK
"""

import torch
from nets.groundstream_I3D import GroundI3D
from nets.groundstream_C3D import GroundC3D
from nets.figurestream_CNN import FigureCNN
from nets.twostream import TwoStream
from ptflops import get_model_complexity_info

from torch.autograd import Variable
import torch.nn.functional as F
from utils.util import *
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

# training function for figure stream
def figure_training(model, num_epochs, dataloaders, optimizer, num_classes,batch_size = 128):
    train_acc_record = []
    val_acc_record = []
    for epoch in range(num_epochs):
        for phase in ['val','val']:
            print(phase)
            labels_counter = {"Corner": 0, "Yellow card": 0, "Clearance": 0, "Throw-in": 0, "Ball out of play": 0,
                              "Substitution": 0}
            correct_counter = {"Corner": 0, "Yellow card": 0, "Clearance": 0, "Throw-in": 0, "Ball out of play": 0,
                               "Substitution": 0}
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
            for index,(inputs,labels) in enumerate(dataloaders[phase]):
                # eliminate none value
                #print(inputs.shape)
                if torch.cuda.is_available():
                    model.cuda()
                label_ = labels

                for i in range(batch_size):
                    labels_counter[onehot_to_str(labels[i])] += 1

                # wrap data in variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                optimizer.zero_grad()
                output = model(inputs)
                loc_loss = F.binary_cross_entropy_with_logits(output, labels)
                #criterion_groups(output, labels.long())
                tot_loc_loss += loc_loss.item()
                # sum up correct label
                for i in range(batch_size):
                    if ((torch.max(output, dim=1)[1][i] == torch.max(labels, dim=1)[1][i])):
                        correct_counter[onehot_to_str(label_[i])] += 1
                    else:
                        print(onehot_to_str(output[i]),onehot_to_str(label_[i]))
                print("counter:",labels_counter)
                correct += sum(torch.max(output, dim=1)[1] == torch.max(labels, dim=1)[1])
                #print(torch.max(output, dim=1)[1] , torch.max(labels, dim=1)[1])
                loss = loc_loss

                tot_loss += loss.item()
                loss.backward()
                optimizer.step()
            # print recognition results
            print("{} {}".format(loc_loss, tot_loss/len(dataloaders[phase].dataset)))
            if phase == "train":
                train_acc_record.append((correct / len(dataloaders[phase].dataset)).cpu().numpy())
            else:
                val_acc_record.append((correct / len(dataloaders[phase].dataset)).cpu().numpy())
            print("correct:",correct, " ",correct / len(dataloaders[phase].dataset))
            # visualize results
            plt.plot(val_acc_record, 'r')
            plt.plot(train_acc_record, 'b')
            plt.savefig('./figure_single.png')
            #torch.save(model.state_dict(), 'best_figure.pt')
            #tot_cls_loss += cls_loss.data[0]

# training function for ground stream
def ground_training(model, num_epochs, dataloaders, optimizer, num_classes=3, batch_size=128):
    train_acc_record = []
    val_acc_record = []
    for epoch in range(num_epochs):

        for phase in ['train','val']:
            print(phase)
            labels_counter = {"Corner": 0, "Yellow card": 0, "Clearance": 0, "Throw-in": 0, "Ball out of play": 0,
                              "Substitution": 0}
            correct_counter = {"Corner": 0, "Yellow card": 0, "Clearance": 0, "Throw-in": 0, "Ball out of play": 0,
                               "Substitution": 0}
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

                if torch.cuda.is_available():
                    model.cuda()
                label_ = labels
                for i in range(2):
                    labels_counter[onehot_to_str(labels[i])] += 1

                # wrap data in variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                """
                flops, params = get_model_complexity_info(model, (3, 50, 320, 320), as_strings=True,
                                                          print_per_layer_stat=True)
                print("|flops: %s |params: %s" % (flops, params))
                """
                labels = Variable(labels.cuda())
                #labels.unsqueeze_(-1)
                #labels = labels.expand(2,num_classes,50)
                optimizer.zero_grad()
                output = model(inputs)

                #output = F.interpolate(output, t, mode='linear')
                loc_loss = F.binary_cross_entropy_with_logits(output, labels)
                tot_loc_loss += loc_loss.item()

                #cls_loss = F.binary_cross_entropy_with_logits(torch.max(output, dim=2)[0], torch.max(labels, dim=2)[0])
                #tot_cls_loss += cls_loss.item()
                # sum up correct label
                for i in range(batch_size):
                    if ((torch.max(output,dim=1)[1][i] == torch.max(labels, dim=1)[1][i])):
                        correct_counter[onehot_to_str(label_[i])] += 1
                        correct += 1
                print("\ncorrect:",correct_counter)
                print("counter:",labels_counter)
                num_steps_per_update = 4
                loss = loc_loss / num_steps_per_update

                tot_loss += loss.item()
                loss.backward()
                optimizer.step()
            if phase == "train":
                train_acc_record.append((correct / len(dataloaders[phase].dataset)))
            else:
                val_acc_record.append((correct / len(dataloaders[phase].dataset)))

            print("{} {}".format(loc_loss, tot_loss/len(dataloaders[phase].dataset)))
            print("correct:",correct, " ",correct / len(dataloaders[phase].dataset))
            # plot accuracy figure
            plt.plot(val_acc_record, 'r')
            plt.plot(train_acc_record, 'b')
            plt.savefig('./c3d_ground_single.png')
            # save model
            torch.save(model.state_dict(), 'best_c3d.pt')

# training function for two stream
def two_stream_training(model,figure_model, ground_model, num_epochs, dataloader, optimizer, num_classes, batch_size=128):
    if model == "I3D":
        ground_stream = GroundI3D(two_stream=True, num_classes=num_classes, in_channels=3)
    elif model == "C3D":
        ground_stream = GroundC3D(two_stream=True, num_classes=num_classes, in_channels=6)
    figure_stream = FigureCNN(num_classes=num_classes, use_pivot_distances=False, two_stream=True)
    # declare model
    ground_stream.load_state_dict(torch.load(ground_model))
    figure_stream.load_state_dict(torch.load(figure_model))
    model = TwoStream(num_classes)
    model.load_state_dict(torch.load("./best_two_stream.pt"))

    for epoch in range(num_epochs):
        for phase in ['train','val']:
            print(phase)
            labels_counter = {"Corner": 0, "Yellow card": 0, "Clearance": 0, "Throw-in": 0, "Ball out of play": 0,
                              "Substitution": 0}
            correct_counter = {"Corner": 0, "Yellow card": 0, "Clearance": 0, "Throw-in": 0, "Ball out of play": 0,
                               "Substitution": 0}

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            tot_loss = 0.0
            for index, (inputs, labels, inputs2, labels2) in enumerate(tqdm(dataloader[phase],
                                                          total=len(dataloader[phase].dataset),
                                                          desc="Processing data:")):
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    model.cuda()
                    ground_stream.cuda()
                    figure_stream.cuda()

                inputs = Variable(inputs.cuda())
                inputs2 = Variable(inputs2.cuda())
                labels = Variable(labels.cuda())
                labels_counter[onehot_to_str(labels)] += 1
                label_ = labels

                ground_output = ground_stream(inputs)
                figure_output = figure_stream(inputs2)
                input = torch.cat((ground_output, figure_output), dim=1)
                """
                flops, params = get_model_complexity_info(model, (2,8192), as_strings=True,
                                                          print_per_layer_stat=True)
                print("|flops: %s |params: %s" % (flops, params))
                """
                output = model(input)
                loss = F.binary_cross_entropy_with_logits(output, labels)
                print(torch.max(labels, dim=1)[1],torch.max(output, dim=1)[1])
                for i in range(batch_size):
                    if torch.max(labels, dim=1)[1][i] == torch.max(output, dim=1)[1][i]:
                        correct_counter[onehot_to_str(label_)] += 1
                print("\ncorrect:", correct_counter)
                print("counter:", labels_counter)
                tot_loss += loss.item()
                loss.backward()
                optimizer.step()
            # save model
            torch.save(model.state_dict(), 'best_two_stream.pt')