import numpy as np, quaternion
import math, copy, random

class GaussianSet():
    def __init__(self, point_cloud_data):  
        self.center = np.empty([len(point_cloud_data), 3])
        self.color = np.empty([len(point_cloud_data), 3])         
        self.spherical_harmonics = np.zeros([len(point_cloud_data), 3, 9])

        i = 0
        for point in point_cloud_data:
            self.center[i] = point_cloud_data[point].xyz
            #print(self.center[i])
            self.color[i] = (point_cloud_data[point].rgb - .5) / 0.28209479177387814
            self.spherical_harmonics[i, :3, 0] = self.color[i]
            i+=1 #bashed my head against the wall for like 10 minutes and it was all ur fault :(

        self.scaling = np.full((len(point_cloud_data), 3), .05)
        self.rotation = np.full((len(point_cloud_data), 4), [1,0,0,0])
        self.opacity = np.full(len(point_cloud_data), -.9542425) # Precomputed inv_sigmoid(0.1)

    def construct_gaussian():
        pass

    # Activation functions & inverse activation functions
    def __sigmoid(x):
        return 1 / (1 + math.exp(-x))
    def __inv_sigmoid(x):
        return math.log(x / (1-x)) 
    def __elu(x, alpha=1.0):
        return x if (x >= 0) else alpha * (math.exp(-x) - 1)
    def __inv_elu(x, alpha=1.0):
        return 1 if (x >= 0) else alpha * math.exp(x)

def train(cameras, images, point_cloud_data, learning_rates, iters=7000):
    gaussians = GaussianSet(point_cloud_data)

    chosen_camera = random.choice(list(images.items()))[1]
    
    camera_r = quaternion.as_rotation_matrix(quaternion.as_quat_array(chosen_camera.qvec))
    camera_t = chosen_camera.tvec

    forward_pass(camera_r, camera_t, gaussians)

    for i in range(iters):
        pass

def forward_pass(camera_r, camera_t, gaussians):
    mod_covariances = np.empty([len(gaussians.center), 3, 3])
    mod_centers = np.empty_like(gaussians.center)

    # this is where we get the covariance & center matrices for each gaussian in coordinance with where the camera is pointing to
    for i in range(len(gaussians.center)):
        rot_matrix = quaternion.as_rotation_matrix(quaternion.as_quat_array(gaussians.rotation[i]))

        cov_matrix =  rot_matrix @ gaussians.scaling[i][:, np.newaxis] @ np.transpose(gaussians.scaling[i][:, np.newaxis]) @ np.transpose(rot_matrix)
        mod_covariances[i] = camera_r @ cov_matrix @ np.transpose(camera_r)
        mod_centers[i] = (camera_r @ gaussians.center[i]) + camera_t
    print(mod_centers[:50])

    
    
