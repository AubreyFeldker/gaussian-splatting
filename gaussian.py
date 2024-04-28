import numpy as np, quaternion, time
from forward_pass import forward_pass
from rasterization import gpu_rasterize, match_gaus_to_tiles
from PIL import Image
from scipy.spatial import KDTree
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

    def __init__(self, centers, shs, scales, rots, opacities, degrees=3):
        self.center = centers
        self.spherical_harmonics = shs
        self.scaling = scales
        self.rotation = rots
        self.opacity = opacities
        self.degrees = degrees

    def rasterize(self, camera, source_image, ctx, queue, program, file_output, result_size=[979,546]):
        t1 = time.perf_counter()
        camera_r = quaternion.as_rotation_matrix(quaternion.as_quat_array(source_image.qvec))
        camera_t = source_image.tvec

        centers, depths, colors, conics, tiles_touched, radii = forward_pass(ctx, queue, program, camera, camera_r, camera_t, self, result_size=result_size, training=False, tile_size=32)
        print("forward pass complete in {time}s".format(time=time.perf_counter()-t1))
        t2 = time.perf_counter()
        key_mapper = match_gaus_to_tiles(ctx, queue, program, tiles_touched, radii, depths, tile_size=32)
        print("key mapping complete in {time}s".format(time=time.perf_counter()-t2))
        t3 = time.perf_counter()
        image = gpu_rasterize(ctx, queue, program, centers, colors, sigmoid(self.opacity), conics, key_mapper, result_size=result_size, training=False, tile_size=32)
        print("rasterization complete in {time}s".format(time=time.perf_counter()-t3))
        Image.fromarray(np.rot90(np.swapaxes(np.uint8(image*255),0,1),2)).save("{output}.jpg".format(output=file_output))

    def save(self, file_path):
        np.savez(file_path,
                     self.center, self.color, self.spherical_harmonics, self.scaling, self.rotation, self.opacity)
        
# Credit to rfeinman on Github for implementation
def knn_distances(points):
    distances, inds = KDTree(points).query(points, k=4)
    return (distances[:, 1:] ** 2).mean(1)