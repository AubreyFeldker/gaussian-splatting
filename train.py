import numpy as np, quaternion
import math, time, random
from scipy.spatial import KDTree
from forward_pass import forward_pass
from backward_pass import backward_pass
from rasterization import rasterize, c_rasterize, gpu_rasterize, match_gaus_to_tiles
from PIL import Image
from functions import *

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

        self.rotation = np.full((len(point_cloud_data), 4), [0.0,0.0,0.0,1.0])
        self.opacity = np.full(len(point_cloud_data), .1)

        self.degrees = 0

    def save(self, file_path):
        np.savez(file_path,
                     self.center, self.color, self.spherical_harmonics, self.scaling, self.rotation, self.opacity)


def train_model(cameras, images, point_cloud_data, learning_rates, ctx = None, queue = None, program = None, iters=2000, result_size=[979,546], t0 = None):
    gaussians = GaussianSet(point_cloud_data)

    
    image_subset = random.sample(list(images.items()), math.floor(len(images) * .2))
    
    print("setup complete in {time}s".format(time=time.perf_counter()-t0))
    for i in range(iters):
        t1 = time.perf_counter()

        image_id = random.randint(0, len(image_subset) - 1)
        source_image = image_subset[image_id][1]
        chosen_camera = cameras[source_image.camera_id]
        
        camera_r = quaternion.as_rotation_matrix(quaternion.as_quat_array(source_image.qvec))
        camera_t = source_image.tvec
        
        centers, depths, colors, conics, clampeds, tiles_touched, radii, ws, ms, ts, covs_2d, covs_3d, dirs = forward_pass(chosen_camera, camera_r, camera_t, gaussians, result_size=result_size, training=True)
        print("forward pass complete in {time}s".format(time=time.perf_counter()-t1))
        t2 = time.perf_counter()
        
        key_mapper = match_gaus_to_tiles(ctx, queue, program, tiles_touched, radii, depths)
        print("key mapping complete in {time}s".format(time=time.perf_counter()-t2))
        t3 = time.perf_counter()
        

        image, d_colors, d_2d_centers, d_conics, d_opacities = gpu_rasterize(ctx, queue, program, centers, colors, gaussians.opacity, conics, key_mapper, result_size=result_size, training=True)
        print("rasterization complete in {time}s".format(time=time.perf_counter()-t3))
        t4 = time.perf_counter()

        d_colors, d_centers, d_shs, d_scales, d_rots = backward_pass(ctx, queue, program, chosen_camera, camera_r, camera_t, gaussians,
                    radii, covs_2d, covs_3d, ws, ms, ts, dirs, clampeds,
                    d_colors, d_2d_centers, d_conics)
        print("backwards pass complete in {time}s".format(time=time.perf_counter()-t4))
        t5 = time.perf_counter()

        pic1 = np.swapaxes(np.uint8(image*255),0,1)
        pic2 = np.array(Image.open(f'./input/images/{image_id+1:06}.jpg'))
        l1_loss = np.mean(np.abs((pic1 - pic2))) / 255

        ep = .000000001
        gaussians.center -= (learning_rates['center'] * np.nan_to_num(d_centers+ep) * l1_loss)
        gaussians.scaling -= (learning_rates['scaling'] * np.nan_to_num(np.log(d_scales+ep)) * l1_loss)
        gaussians.spherical_harmonics -= (learning_rates['sh'] * np.nan_to_num(d_shs+ep) * l1_loss)
        gaussians.rotation -= (learning_rates['rotation'] * np.nan_to_num(d_rots+ep) * l1_loss)
        gaussians.opacity -= (learning_rates['opacity'] * d_opacities+ep * l1_loss)

        print("backwards training complete in {time}s".format(time=time.perf_counter()-t5))
        
        print("pass {iter} completed in {time}s\n-----------------------".format(time=time.perf_counter()-t1, iter=i))
        if(i % 50 == 0):
            Image.fromarray(pic1).save("output/test_run/pass_{iter}.jpg".format(iter=i))
        if(i % 100 == 0):
            gaussians.save('output/test_run/iter_{iter}_gaus'.format(iter=i))

#image = c_rasterize(centers, colors, gaussians.opacity, conics, key_mapper, result_size=result_size)
#image = rasterize(centers, colors, gaussians.opacity, conics, key_mapper, result_size=result_size)

# Credit to rfeinman on Github for implementation
def knn_distances(points):
    distances, inds = KDTree(points).query(points, k=4)
    return (distances[:, 1:] ** 2).mean(1)

