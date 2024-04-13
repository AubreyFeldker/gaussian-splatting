import torch, torch.optim as optim, torch.nn as nn
import numpy as np

class Gaussians():
    def __init__(self, var):
        self.center = torch.empty(0)
        self.color = torch.empty(0)
        self.scaling = torch.empty(0)
        self.rotation = torch.empty(0)
        self.opacity = torch.empty(0)
        self.spherical_harmonics = torch.empty(0)

def inv_sigmoid(x):
    return torch.log(x / (1-x))     
 
def gaussian_setup(point_cloud_data):
    gaussians = Gaussians(0)
    point_centers = []
    point_colors = []

    for point in point_cloud_data:
        point_centers.append(point_cloud_data[point].xyz)
        point_colors.append((point_cloud_data[point].rgb - .5) / 0.28209479177387814)

    gaus_centers = torch.tensor(point_centers)
    gaus_colors = torch.tensor(point_colors)

    gaus_scales = torch.full((gaus_centers.shape[0], 3), .05)
    gaus_rotations = torch.zeros((gaus_centers.shape[0], 4))
    gaus_rotations[:, 0] = 1

    gaus_opacities = inv_sigmoid(.1 * torch.ones((gaus_centers.shape[0], 1), dtype=torch.float))

    gaus_shs = torch.zeros((gaus_colors.shape[0], 3, 9), dtype=torch.float)
    gaus_shs[:, :3, 0] = gaus_colors
    #gaus_shs[:, 3:, 1:] = 0.0

    # Turn the tensors into specified neural network parameters
    gaussians.color = gaus_colors

    gaussians.center = nn.Parameter(gaus_centers.requires_grad_(True))
    gaussians.scaling = nn.Parameter(gaus_scales.requires_grad_(True))
    gaussians.rotation = nn.Parameter(gaus_rotations.requires_grad_(True))
    gaussians.opacity = nn.Parameter(gaus_opacities.requires_grad_(True))
    gaussians.spherical_harmonics = nn.Parameter(gaus_shs.requires_grad_(True))

    print(gaussians.center.shape)

    return gaussians

def train(point_cloud_data):
    gaussians = gaussian_setup(point_cloud_data)

    model_setup = [
        {'params': [gaussians.center], 'lr': .00016, "name": "center"},
        {'params': [gaussians.spherical_harmonics], 'lr': .0025, "name": "sh_features"},
        {'params': [gaussians.scaling], 'lr': .005, "name": "scaling"},
        {'params': [gaussians.opacity], 'lr': .05, "name": "opacity"},
        {'params': [gaussians.rotation], 'lr': .001, "name": "rotation"}
    ]
    
    optimizer = optim.Adam(model_setup, lr=0.0)