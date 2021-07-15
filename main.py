import sys

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2, os

sys.path.append("F:\Disssertation\single_test\detectron2")

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from single_test.utils.util import *

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




def get_all_action(path,action="Direct free-kick",label="Labels-v2.json"):
    """

    :param path:
    :param action:
    :param label:
    :return:
    """
    actions = []
    all_folder = os.listdir(path)
    config = configparser.ConfigParser()



    for game in all_folder:
        file_path = os.path.join(path,game,label)
        with open(file_path) as f:
            data = json.load(f)

        for element in data["annotations"]:
            if element["gameTime"][0] == "1":
                current_game = "1_HQ.mkv"
            else:
                current_game = "2_HQ.mkv"
            config.read(os.path.join(path, game, "video.ini"))
            start_frame = config.get(current_game, "start_time_second")
            if element["label"] == action and element["visibility"]=="visible":
                # store all action positions and game name
                actions.append([os.path.join(path,game),
                                element["gameTime"],
                                element["position"],
                                float(start_frame)-1])
    return actions

def show_action(all_action):
    for video in all_action:
        frame_num = int(video[2])

        cap = cv2.VideoCapture(os.path.join(video[0],video[1][0]+"_HQ.mkv"))
        fps = cap.get(cv2.CAP_PROP_FPS)

        start_frame = int(video[3]) * fps
        cap.set(1,start_frame+frame_num/(1000/fps)-10)

        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)

        ret, frame = cap.read()
        current = 0
        while cap.isOpened():
            current = current+1

            outputs = predictor(frame)
            bbox = outputs["instances"].pred_boxes.tensor
            get_skeleton(bbox,frame,cfg)
            v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow('window', out[:, :, ::-1])
            success, frame = cap.read()
            if current >= fps*2:
                break
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break




def get_skeleton(bbox,frame,cfg,offset=10):
    print(len(bbox))

    for index in range(len(bbox)):
        x1,y1,x2,y2 = bbox[index].cpu().numpy().astype(int)
        frame_ = frame[y1-offset:y2+offset, x1-offset:x2+offset, :]

        cv2.imshow("",frame_)
        scale_percent = 10
        width = int(frame_.shape[1] * scale_percent / 10)
        height = int(frame_.shape[0] * scale_percent / 10)
        dim = (width, height)
        src_img = cv2.resize(frame_, dim, interpolation=cv2.INTER_AREA)

        datum.cvInputData = src_img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        #print(datum.poseKeypoints[0,:,:])
        if datum.poseKeypoints is not None:
            cv2.imwrite("./results/test" + str(index) + ".jpg",datum.cvOutputData)





        frame_ = frame[y1-offset:y2+offset, x1-offset:x2+offset, :]
        outputs = predictor(frame_)
        
        #cv2.rectangle(frame, (x1-offset, y1-offset), (x2+offset, y2+offset), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)

        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('window', out.get_image()[:, :, ::-1])

    return 0

#def random_colour_bbox():



path = r"F:\Disssertation\soccernet data\rawvideo\england_epl\2015-2016"

dataset = ["england_epl"]


all_actions = get_all_action(path)
show_action(all_actions)