from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.utils.data as data_utl
from PIL import Image
import cv2, json, os, pickle
from tqdm import tqdm
import numpy as np
import utils.util as utils
from get_skeleton import show_action
class Video_to_seq(data_utl.Dataset):
    def __init__(self, data, root, output, transforms=None, threshold=4, shown=False, target_action=["Clearance","Throw-in"]):
        self.data = self.make_dataset(data,root,num_classes=21,num_sec=threshold)
        self.split_file = data
        self.output = output
        self.transforms = transforms
        self.root = root
        self.shown = shown
        self.frames = []
        self.torch_data = []
        for index, video in tqdm(enumerate(self.data),total= len(self.data), desc ="Convert video to image sequences:"):
            vid, label, start, action_frame,num_sec = video
            print(utils.onehot_to_str(label), target_action)

            if utils.onehot_to_str(label) in target_action:
                self.load_frames(vid, start, action_frame, label, num_sec, index)

    def video_to_tensor(self,img):
        """Convert a ``numpy.ndarray`` to tensor.
        Converts a numpy.ndarray (T x H x W x C)
        to a torch.FloatTensor of shape (C x T x H x W)

        Args:
             pic (numpy.ndarray): Video to be converted to tensor.
        Returns:
             Tensor: Converted video.
        """
        return torch.from_numpy(img.transpose([3, 0, 1, 2]))

    def make_dataset(self, data, root, num_classes=17, num_sec=4):
        dataset = []
        if not os.path.exists(data):
            utils.creaet_data_json(root,data)

        with open(data, 'r') as f:
            data = json.load(f)
        for vid in data:
            label_ = vid["label"]
            video = vid["video"]
            start = vid["start_time"]
            action_frame = vid["action_frame"]

            label = utils.index_to_onehot(label_, num_classes)
            dataset.append((video,label,start,action_frame,num_sec))
        return dataset

    def load_frames(self, video_path, start_time,action_frame, label, num_sec,index):
        cap = cv2.VideoCapture(os.path.join(self.root,video_path))
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # get video frame rate per sec
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = float(start_time) * fps
        start_frame = start_frame + (float(action_frame) / 1000) * fps - fps
        current_frame = 1
        label_ = utils.onehot_to_str(label)
        # evenly sample all action frames to 100 frames
        evenly_frame = np.linspace(start_frame, start_frame+num_sec*fps, 100,dtype = int)
        while cap.isOpened():
            for i in evenly_frame:
                # set frame to when action occur
                cap.set(1, i)
                current_frame += 1
                success, frame = cap.read()
                #frame = cv2.resize(frame, dsize=(640, 360))
                # whether display current processing frame?
                if self.shown:
                    cv2.imshow(label_, frame)
                if not os.path.exists("./data/videoData/{}/{}".format(self.output,index)):
                    os.makedirs("./data/videoData/{}/{}".format(self.output,index))
                cv2.imwrite("./data/videoData/{}/{}/{}_{}.jpg".format(self.output,index,label_,current_frame),frame)
                self.write_to_txt("{} {} {}".format(index,label_,current_frame))

                if cv2.waitKey(25) & 0xFF == ord('q') or current_frame > num_sec*fps:
                    break
                else:
                    continue
            break
            cv2.destroyWindow(label_)

        return np.asarray(frame, dtype=np.float32)
    def write_to_txt(self,data):
        with open("./data/VideoData/seq_data3.txt","a") as f:
            f.write(data+'\n')
        return data
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        return self.torch_data

    def __len__(self):
        return len(self.data)





