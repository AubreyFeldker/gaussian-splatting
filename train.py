import numpy as np, quaternion
import math, copy, random, ctypes
from scipy.spatial import KDTree
import forward_pass as forward, rasterization
from PIL import Image

class GaussianSet():
    def __init__(self, point_cloud_data):  
        self.center = np.empty([len(point_cloud_data), 3])
        self.color = np.empty([len(point_cloud_data), 3])         
        self.spherical_harmonics = np.zeros([len(point_cloud_data), 16, 3])

        i = 0
        for point in point_cloud_data:
            self.center[i] = point_cloud_data[point].xyz
            #print(self.center[i])
            self.color[i] = (point_cloud_data[point].rgb - .5) / 0.28209479177387814
            self.spherical_harmonics[i, 0] = self.color[i]
            i+=1 #bashed my head against the wall for like 10 minutes and it was all ur fault :(

        # Get initial gaussian scaling based on initial point cloud clustering distances
        distances = np.clip(knn_distances(self.center), a_min=.0000001, a_max=None)
        self.scaling = np.repeat(distances, 3).reshape((len(point_cloud_data), 3)) 

        self.rotation = np.full((len(point_cloud_data), 4), [1,0,0,0])
        self.opacity = np.full(len(point_cloud_data), .1) # Precomputed inv_sigmoid(0.1)

        self.degrees = 0

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

def train_model(cameras, images, point_cloud_data, learning_rates, iters=7000):
    gaussians = GaussianSet(point_cloud_data)

    #source_image = random.choice(list(images.items()))[1]
    source_image = list(images.items())[0][1]
    chosen_camera = cameras[source_image.camera_id]
    print(source_image)
    
    camera_r = quaternion.as_rotation_matrix(quaternion.as_quat_array(source_image.qvec))
    camera_t = source_image.tvec

    centers, depths, colors, conics, clampeds, tiles_touched, radii = forward.forward_pass(chosen_camera, camera_r, camera_t, gaussians)
    print("key mapping time")
    key_mapper = rasterization.match_gaus_to_tiles(tiles_touched, radii, depths)
    print("rasterization time")
    image = rasterization.rasterize(centers, colors, gaussians.opacity, conics, key_mapper)

    Image.fromarray(np.swapaxes(np.uint8(image*255),0,1)).save("output/result_2.jpg")

# Credit to rfeinman on Github for implementation
def knn_distances(points):
    distances, inds = KDTree(points).query(points, k=4)
    return (distances[:, 1:] ** 2).mean(1)
