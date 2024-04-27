import numpy as np, quaternion
import math, time
from scipy.spatial import KDTree
from forward_pass import forward_pass
from backward_pass import backward_pass
from rasterization import rasterize, c_rasterize, gpu_rasterize, match_gaus_to_tiles
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
        self.scaling = np.repeat(np.log(distances), 3).reshape((len(point_cloud_data), 3)) 

        self.rotation = np.full((len(point_cloud_data), 4), [0,0,0,1])
        #TODO: computer proper sigmoidation
        self.opacity = np.full(len(point_cloud_data), .1) # Precomputed inv_sigmoid(0.1)

        self.degrees = 0

def train_model(cameras, images, point_cloud_data, learning_rates, ctx = None, queue = None, program = None, iters=7000, result_size=[979,546], t0 = None):
    gaussians = GaussianSet(point_cloud_data)

    #source_image = random.choice(list(images.items()))[1]
    source_image = list(images.items())[0][1]
    chosen_camera = cameras[source_image.camera_id]
    #print(source_image)
    
    camera_r = quaternion.as_rotation_matrix(quaternion.as_quat_array(source_image.qvec))
    camera_t = source_image.tvec

    
    print("setup complete in {time}s".format(time=time.perf_counter()-t0))
    t1 = time.perf_counter()
    
    centers, depths, colors, conics, clampeds, tiles_touched, radii, ws, ts, covs_2d, covs_3d = forward_pass(chosen_camera, camera_r, camera_t, gaussians, result_size=result_size, training=True)
    print("forward pass complete in {time}s".format(time=time.perf_counter()-t1))
    t1 = time.perf_counter()
    
    key_mapper = match_gaus_to_tiles(ctx, queue, program, tiles_touched, radii, depths)
    t2 = time.perf_counter()
    print("key mapping complete in {time}s".format(time=t2-t1))
    

    image, d_colors, d_2d_centers, d_conics, d_opacity = gpu_rasterize(ctx, queue, program, centers, colors, gaussians.opacity, conics, key_mapper, result_size=result_size, training=True)
    #image = c_rasterize(centers, colors, gaussians.opacity, conics, key_mapper, result_size=result_size)
    #image = rasterize(centers, colors, gaussians.opacity, conics, key_mapper, result_size=result_size)
    print("rasterization complete in {time}s".format(time=time.perf_counter()-t2))
    t3 = time.perf_counter()

    backward_pass(ctx, queue, program, chosen_camera, camera_r, camera_t, gaussians,
                  radii, covs_2d, covs_3d, ws, ts,
                  d_colors, d_2d_centers, d_conics, d_opacity)
    print("backwards pass complete in {time}s".format(time=time.perf_counter()-t3))
    Image.fromarray(np.swapaxes(np.uint8(image*255),0,1)).save("output/result_9.jpg")

# Credit to rfeinman on Github for implementation
def knn_distances(points):
    distances, inds = KDTree(points).query(points, k=4)
    return (distances[:, 1:] ** 2).mean(1)

# Activation functions & inverse activation functions
def __sigmoid(x):
    return 1 / (1 + math.exp(-x))
def __inv_sigmoid(x):
    return math.log(x / (1-x)) 
def __elu(x, alpha=1.0):
    return x if (x >= 0) else alpha * (math.exp(-x) - 1)
def __inv_elu(x, alpha=1.0):
    return 1 if (x >= 0) else alpha * math.exp(x)