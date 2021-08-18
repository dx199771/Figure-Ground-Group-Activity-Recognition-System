# Two-Stream Figure–Ground Group Activity Recognition System in Soccer Videos
In this reporsitory, we proposed a two-stream figure--ground group activity recognition system in soccer games. The system contains two streams:
1) Ground stream: mainly 

![](/assets/ss.jpg)
> Human visual system tends to percieve the world by figure-ground perception


## Network structure
![](/assets/draft new.png)

## Dependencies
```
numpy==1.20.3
seaborn==0.11.1
pandas==1.2.4
opencv_python==4.5.2.52
torch==1.8.1+cu111
matplotlib==3.3.4
detectron2.egg==info
ptflops==0.6.6
torchvision==0.9.1+cu111
tqdm==4.59.0
net==0.1
Pillow==8.3.1
Facebook detectron2
OpenPose
```

## Datasets
If you want to use SoccerNet-v2 dataset, you need to access the [official website](https://soccer-net.org/) and fill in the [form](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform) to request dataset access.

After you acquired the access for the dataset, download the HQ resolution dataset (you can choose the leagues you like), and download the label file.

Unzip the soccer game videos and labels then put them together in the data folder, Be noticed that in each game video folder there are three .json file. We only need the json file with soocer game action label.

## Getting Started
0. Stage0: Download CMake-GUI (You can either download "Windows x64 Installer" or "Windows x64 ZIP").
1. Stage1. setup Facebook detectron2 for player detection, instructions can be found at: [FaceBook Detectron2](https://github.com/facebookresearch/detectron2)
2. Stage2: setup OpenPose for skeleton detection, instructions can be found at: [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
3. Stage3: Make sure you put the downloaded data in data folder, change the root parameter in line 56 in [main.py](main.py) file and run the code.
4. Stage4: Comment out the last few lines in [dataloader.py](/utils/dataloader.py) and run dataloder.py until finished (this is for extract skeleton features).


## License
Two-Stream Figure–Ground Group Activity Recognition System in Soccer Videos is released under the [GNU General Public License v3.0](LICENSE)

##Citation
```

```
