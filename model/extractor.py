import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import vgg16


def customizedVGG16():
    model = vgg16(pretrained=True)
    
    features = list(model.features)[:30]
    classifier = model.classifier
    
    classifier = list(classifier)
    # delete the Linear layer
    del classifier[6]
    classifier = nn.Sequential(*classifier)

    #freeze top4 conv layer
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    features = nn.Sequential(*features)
        
    return features, classifier

def extract_feature_map(extractor, sample):
    """
    Input: samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
    Output: a cuda tensor with size [batch_size, 512, 16, 114]
    """
    batch_size = sample.shape[0]
    extracted_fm = []
    for i in range(batch_size):
        cur_sample = sample[i]
        # extracted_sample = [512, 16, 19]
        extracted_sample = extractor(cur_sample)
        tensor_tuples = torch.unbind(extracted_sample, dim=0)
        concatenated_fm = torch.cat(tensor_tuples, dim=2)
        extracted_fm.append(concatenated_fm)
    return torch.stack(extracted_fm)

    