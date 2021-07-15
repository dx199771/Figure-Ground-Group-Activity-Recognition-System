from tqdm import tqdm
import os, sys, json
import numpy as np
from sys import platform
import configparser

def get_all_game_path(dataset_path, all_dataset_path=[]):
    all_folder = os.listdir(dataset_path)
    for item in all_folder:
        season_path = os.path.join(dataset_path, item)
        all_season_folder = os.listdir(season_path)
        for game in all_season_folder:
            game_path = os.path.join(season_path,game)
            all_dataset_path.append(game_path)
    return all_dataset_path


def get_file_list(root_dir = "", extensions = []):
    file_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return file_list

def creaet_data_json(path,opt):
    num_frame = 50
    config = configparser.ConfigParser()
    file_list = get_file_list(path,["Labels-v2.json"])
    data_ = []
    for i in file_list:
        with open(i,'r') as f:
            data = json.load(f)
        url = data['UrlLocal']
        annotations = data["annotations"]
        head = os.path.join(path, url)
        for anno in annotations:
            if anno["visibility"] == "visible":
                label = anno["label"]
                position = anno["position"]
                current_game = anno["gameTime"][0]
                if current_game == "1":
                    current_game = "1_HQ.mkv"
                else:
                    current_game = "2_HQ.mkv"
                config.read(os.path.join(head, "video.ini"))
                start_frame = config.get(current_game, "start_time_second")

                # create new json format and save it
                x = {
                    "video": os.path.join(url,current_game),
                    "start_time": start_frame,
                    "label": label,
                    "action_frame": position,
                    "num_frame": num_frame,
                }
                data_.append(x)
        json_str = json.dumps(data_, indent=4)

        with open(opt, 'w') as f:
            f.write(json_str)

def label_lookup(label):
    """
    labels = ["Foul", "Yellow card","Red card","Yellow->red card", "Penalty", "Goal",
              "Direct free-kick","Indirect free-kick", "Shots on target","Shots off target", "Throw-in",
              "Clearance", "Ball out of play", "Substitution", "Corner", "Kick-off", "Offside"]
    """
    labels = ["Corner", "Yellow card", "Clearance","Throw-in","Ball out of play","Substitution"]

    return labels.index(label)

def index_to_onehot(label_,num_classes):
    index = label_lookup(label_)
    label = np.zeros(num_classes, np.float32)
    label[index] = 1
    return label

def label_to_str(idx):
    """
    labels = ["Foul", "Yellow card","Red card","Yellow->red card", "Penalty", "Goal",
              "Direct free-kick","Indirect free-kick", "Shots on target","Shots off target", "Throw-in",
              "Clearance", "Ball out of play", "Substitution", "Corner", "Kick-off", "Offside"]
    """
    labels = ["Corner", "Yellow card", "Clearance","Throw-in","Ball out of play","Substitution"]

    return labels[idx]

def onehot_to_str(array):
    return label_to_str(array.argmax())

def onehot_to_index(array):
    return array.argmax(axis=1)

