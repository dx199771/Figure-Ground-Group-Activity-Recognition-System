# abandon file, use get_skeleton.py instead

from single_test.utils.util import *
import configparser
import json
from single_test.utils.video2seq import Video_dataloader
from single_test.nets.I3D import *
import torchvision.models as models

class Feature_extractor():
    def __init__(self,
                 data_path="",
                 extractor="ResNet",
                 target_action=[]
                 ):
        self.data_path = data_path
        self.extractor = extractor
        self.target_action = target_action
    def extract_all_games(self,data_path):
        all_game_path = get_all_game_path(data_path)
        for game in all_game_path:
            #try:
                self.extract_feature(game)
            #except:
            #    print("No Game Found!")
    def extract_feature(self,game_path):

        # read configuration file
        config = configparser.ConfigParser()
        config.read(os.path.join(game_path, "video.ini"))

        for video in ["1_HQ.mkv","2_HQ.mkv"]:
            start_frame = config.get(video, "start_time_second")
            video_path = os.path.join(game_path,video)


            file_path = os.path.join(game_path,"Labels-v2.json")
            with open(file_path) as f:
                data = json.load(f)
            for element in data["annotations"]:
                if element["gameTime"][0] == video[0] and element["visibility"] == "visible":
                    action = element["label"]
                    if action in self.target_action:
                        position = element["position"]
                        data_loader = Video_dataloader(video_path,
                                                       thres=150,
                                                       resize=(1280,720),
                                                       start_time=start_frame,
                                                       action_frame=position,
                                                       action=action,
                                                       shown=True)
                        data = data_loader.read_video()
                        if self.extractor == "ResNet":
                            self.extract_resnet(data,model_name="resnet152")
                        elif self.extractor == "I3D":
                            self.extract_I3D(data)
        return data



    def extract_resnet(self, data, model_name):
        extract_list = ["layer3"]
        model = models.resnet.__dict__["resnet152"](pretrained=True)
        extract_result = Resnet_singlelayer(model, extract_list)

        data = torch.stack(data["image"])
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = data.to('cuda')
            single_feature = extract_result(input_batch.cpu())[0]
            print(np.array(single_feature.detach().numpy()).shape)
            model.to('cuda')
        with torch.no_grad():
            output = model(input_batch)

        return output

    def extract_I3D(self, data):
        data = torch.stack(data["image"])
        model = I3D(5, in_channels=3)
        # load pre-trained imagenet model
        #model.load_state_dict(torch.load('models/flow_imagenet.pt'))
        if torch.cuda.is_available():
            input_batch = data.to('cuda')
            model.to('cuda')
        with torch.no_grad():
            input_batch = input_batch.permute(1, 0, 2, 3)
            input_batch = torch.unsqueeze(input_batch, 0)

            output = model(input_batch)
            print(output.cpu().numpy().shape)

class Resnet_singlelayer(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(Resnet_singlelayer, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():

            if name == "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


all_dataset_path = []

dataset_path = r"/soccernet data/rawvideo/england_epl"


feature_extractor = Feature_extractor(data_path=dataset_path,
                                      extractor="I3D",
                                      target_action=["Clearance"]
                                      )

feature_extractor.extract_all_games(data_path=dataset_path)

