"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os

# import your model class
# import ...

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

def reorder_coord(pred_bboxes):
    xmin, ymin, xmax, ymax = pred_bboxes.unbind(1)
    return torch.stack((xmax, xmax, xmin, xmin, ymax, ymin, ymax, ymin), dim=1).view(-1, 2, 4)

def concat_features(features, dim = 2):
    #dim 0 ==> stacking the images in the channel dimension
    #dim 1 ==> stacking the images in row dimension
    #dim 2 ==> stacking the images in column dimension
    tensor_tuples = torch.unbind(features, dim=0)
    concatenated_fm = torch.cat(tensor_tuples, dim=dim)
    return concatenated_fm

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

class ModelLoader():
    # Fill the information for your team
    team_name = 'team_name'
    round_number = 1
    team_member = []
    contact_email = '@nyu.edu'

    def __init__(self, model_file='put_your_model_file_name_here'):
        """
        model_file = {'get_bboxes_model': object_detection_model_path,
                      'get_binary_RM_model': None}
        """
        
        self.bboxes_model_file =  model_file['get_bboxes_model']

        self.bboxes_checkpoint = torch.load(self.bboxes_model_file)
        self.bboxes_feature_extractor_weights = self.bboxes_checkpoint['feature_extractor']
        self.bboxes_model_weights = self.bboxes_checkpoint['fasterRCNN']
        

        # Initiate the feature extractor
        # Get model
        self.bboxes_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                                 pretrained_backbone=False)
        self.bboxes_model = self.bboxes_model.to(device)
        self.bboxes_model.load_state_dict(self.bboxes_model_weights)
        
        # Get feature extractor
        self.bboxes_feature_extractor = torchvision.models.resnet18(pretrained = False)
        self.bboxes_feature_extractor = nn.Sequential(*list(self.bboxes_feature_extractor.children())[:-3])
        self.bboxes_feature_extractor.to(device)
        self.bboxes_feature_extractor.load_state_dict(self.bboxes_feature_extractor_weights)


        self.bboxes_model.eval()
        self.bboxes_feature_extractor.eval()
    
    
    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        #Preparing inputs
        batchsize = samples.shape[0]
        fe_batch = []
        for i in range(batchsize):
            image_tensor = samples[i]
            features = self.bboxes_feature_extractor(image_tensor)
            #print(features.shape)
            features = concat_features(features)
            features = features.view(3, 256, 40*16)
            #print(features.shape)
            fe_batch.append(features)

        images = list(image.to(device) for image in fe_batch)
        predictions = self.bboxes_model(images)
        res = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            pred_bboxes = prediction['boxes']
            reorder_pred_bboxes = reorder_coord(pred_bboxes)
            res.append(reorder_pred_bboxes)

        return res

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        
        return torch.rand(1, 800, 800) > 0.5