import json
import os
import sys
from collections import defaultdict

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2
from random import randint

sys.path.append("F:\Disssertation\single_test\detectron2")

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from utils.util import *

dir_path = os.path.dirname(os.path.realpath(__file__))

# Import Openpose (Windows/Ubuntu/OSX)
print("openpose directroy: " + dir_path)
try:
    # Windows Import
    if platform == "win32":
        dir_path = "./build"
        # Change these variables to point to the correct folder (Release/x64 etc.)
        #sys.path.append(os.path.join(dir_path,'build/python/openpose/Release'));
        sys.path.append(dir_path+'./openpose/Release');
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/x64/Release;' + dir_path + '/bin;'
        import pyopenpose as op

    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('./openpose/python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu),
        # you can also access the OpenPose/python module from there.
        # This will install OpenPose and the python library at your desired installation path.
        # Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        print("imported")
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. '
          'Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e
# initialize openpose API

params = dict()
params["model_folder"] = "./build/models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def show_action(action_folder,rootpath):
    path = os.path.join(action_folder)
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)



    file_list = os.listdir(action_folder)
    skeleton_data = []
    frame = cv2.imread(os.path.join(path, file_list[int(len(file_list)/2)]))
    #frame = cv2.imread(os.path.join(path, file_list[15]))
    outputs = predictor(frame)

    #v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=2)
    #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow('window', out.get_image()[:, :, ::-1])
    #cv2.imwrite("ss.jpg",out.get_image()[:, :, ::-1])


    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    for i, newbox in enumerate(boxes):

        skeleton_data_ = get_skeleton(newbox, frame, i)
        skeleton_data.append(np.array(skeleton_data_))
    #print(rootpath, i)
    frame_skeleton = np.array(skeleton_data)
    with open('./data/SkeletonData/{}.npy'.format(rootpath), 'wb') as fp:
        np.save(fp, frame_skeleton)
    #TODO tracked multiframes feature extraction
    """
    sortedfiles = sorted(file_list, key=lambda x:int(x.split("_")[1][:-4]))
    skeleton_data = {}
    object_missing = True
    frame_hog = []
    for index, name in enumerate(sortedfiles):
        hog = cv2.HOGDescriptor()
        frame = cv2.imread(os.path.join(path,name))
        h = hog.compute(frame)
        frame_hog.append(np.mean(h*100))
        current = sum(frame_hog) / len(frame_hog)
        print(current,  np.mean(h*100), abs(current - np.mean(h*100)))
        if index == 0 or abs(current - np.mean(h*100))>1.5:
            frame_hog = []
            multiTracker = cv2.legacy.MultiTracker_create()

            outputs = predictor(frame)
            bbox = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            for object in bbox:
                #print(object)
                xmin, ymin = object[0],object[1]
                boxwidth, boxheight= object[2]-object[0], object[3]-object[1]
                object = (xmin,ymin,boxwidth,boxheight)
                multiTracker.add(inital_tracker("CSRT"), frame, object)


        #print(np.mean(h*100),os.path.join(path,name))
        object_missing, boxes = multiTracker.update(frame)
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, (randint(0, 255), randint(0, 255), randint(0, 255)), 2, 1)
            #print(newbox.shape,frame.shape,i)

            skeleton_single = get_skeleton(newbox, frame, i, index, skeleton_data)
            if skeleton_single is not None:
                if i not in skeleton_data:
                    skeleton_data[i] = []

                skeleton_data[i].append(skeleton_single)
                #skeleton_data.append({i:skeleton_single})
            #print(skeleton_data)

        cv2.imshow('MultiTracker', frame)

        #print(skeleton_data)
        #v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2.imshow('window', out.get_image()[:, :, ::-1])

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    with open('./data/SkeletonData/{}.json'.format(rootpath), 'w') as fp:
        json.dump(skeleton_data,fp)
    """
    return 0

def inital_tracker(tracker_type="CSRT"):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.legacy.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.legacy.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()

    return tracker

def get_skeleton(bbox,frame, player_index, offset=15):
    x1,y1,x2,y2 = bbox.astype(int)
    height, width, _  = frame.shape

    if y1-offset>=0 and x1-offset>=0 and y2+offset<height and x2+offset<width:
        frame_ = frame[y1-offset:(y2)+offset,x1-offset:x2+offset, :]
    else:
        frame_ = frame[y1:y2,x1:x2, :]
    #cv2.imshow("",frame_)
    scale_percent = 8
    width = int(frame_.shape[1] * scale_percent / 10)
    height = int(frame_.shape[0] * scale_percent / 10)
    dim = (width, height)
    src_img = cv2.resize(frame_, dim, interpolation=cv2.INTER_AREA)
    datum.cvInputData = src_img
    #print(datum.cvOutputData.shape)
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    #if datum.poseKeypoints is not None: print( datum.poseKeypoints[0,:,:])

    if datum.poseKeypoints is not None:
        cv2.imwrite("./results/test" + str(player_index) +"_"+ str("ss") + ".jpg",datum.cvOutputData)
        #print(player_index, frame_index,datum.poseKeypoints[0,:,:] )

        keypoints = datum.poseKeypoints[0,:,:].tolist()
        return keypoints
for i in os.listdir(r"F:\Disssertation\single_test\data\VideoData\seq3"):
    target_folder = os.path.join(r"F:\Disssertation\single_test\data\VideoData\seq3",i)
    show_action(target_folder,"seq3/"+i)
for i in os.listdir(r"F:\Disssertation\single_test\data\VideoData\seq4"):
    target_folder = os.path.join(r"F:\Disssertation\single_test\data\VideoData\seq4",i)
    show_action(target_folder,"seq4/"+i)
"""
all_keys = {}
with open("./data/1436.json", 'r') as r:
    data = json.load(r)
    for i in data:
        key = list(i.keys())[0]
        all_keys[key] = i[key]

        print(i)
"""