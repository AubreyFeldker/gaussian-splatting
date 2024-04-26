import numpy as np, math, copy
from ctypes import *
import pyopencl as cl

def match_gaus_to_tiles(tiles_touched, radiis, depths, result_size=[979,546], tile_size=16):
    hor_tiles = math.ceil(result_size[0] * 1.0 / tile_size)
    ver_tiles = math.ceil(result_size[1] * 1.0 / tile_size)
    key_mapper = [[dict() for i in range(ver_tiles)] for j in range(hor_tiles)]

    for gaus in range(len(tiles_touched)):
        if (radiis[gaus] == 0):
            continue

        for i in range(tiles_touched[gaus,0,0], tiles_touched[gaus,1,0]):
            for j in range(tiles_touched[gaus,0,1], tiles_touched[gaus,1,1]):
                key_mapper[i][j][depths[gaus]] = gaus

    return key_mapper

def gpu_rasterize(ctx, queue, program, centers: np.ndarray, colors: np.ndarray, opacities: np.ndarray, conics: np.ndarray, mapped_keys,
              training=True, result_size=[979,546], tile_size=16, background_color=np.zeros(3)):
    image = np.zeros([result_size[0], result_size[1], 3])
    image_chunk = np.zeros([tile_size, tile_size, 3])
    
    num_gaus = len(centers)
    d_colors = np.zeros([num_gaus, 3])
    d_centers = np.zeros([num_gaus, 2])
    d_conics = np.zeros([num_gaus, 3])
    d_opacity = np.zeros(num_gaus)

    other_data = np.empty(6, dtype=np.int32)
    other_data[1] = 1 if (training) else 0
    other_data[4] = result_size[0]
    other_data[5] = result_size[1]

    mf = cl.mem_flags
    centers_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=centers.flatten())
    colors_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=colors.flatten())
    opacities_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=opacities)
    conics_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=conics.flatten())
    background_color_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=background_color)

    image_g = cl.Buffer(ctx, mf.WRITE_ONLY, image_chunk.nbytes)
    d_colors_g = cl.Buffer(ctx, mf.WRITE_ONLY, d_colors.nbytes)
    d_centers_g = cl.Buffer(ctx, mf.WRITE_ONLY, d_centers.nbytes)
    d_conics_g = cl.Buffer(ctx, mf.WRITE_ONLY, d_conics.nbytes)
    d_opacity_g = cl.Buffer(ctx, mf.WRITE_ONLY, d_opacity.nbytes)

    hor_tiles = math.ceil(result_size[0] / tile_size)
    ver_tiles = math.ceil(result_size[1] / tile_size)

    for i in range(hor_tiles):
        for j in range(ver_tiles):
            depth_sorted = np.asarray(sorted(mapped_keys[i][j].items()), dtype=np.int32)
            if (len(depth_sorted > 0)):
                depth_sorted_gaus = np.ascontiguousarray(depth_sorted[...,1]) # flatten the dictionary to a depth-sorted 1D array
                other_data[0] = len(depth_sorted_gaus)
            else:
                depth_sorted_gaus = np.zeros(1, dtype=np.int32)
                other_data[0] = 0
       
            other_data[2] = i * tile_size
            other_data[3] = j * tile_size

            gaus_stack_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=depth_sorted_gaus)
            other_data_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=other_data)

            rast_kernel = program.rasterize
            rast_kernel(queue, np.arange(16*16).shape, None,
                              centers_g, colors_g, opacities_g, conics_g, gaus_stack_g, background_color_g, other_data_g,
                              image_g, d_colors_g, d_centers_g, d_conics_g, d_opacity_g)
            
            last_x = min((i+1)*tile_size,result_size[0])
            last_y = min((j+1)*tile_size,result_size[1])
            chunk_x = min(tile_size, result_size[0]-i*tile_size)
            chunk_y = min(tile_size, result_size[1]-j*tile_size)

            cl.enqueue_copy(queue, image_chunk, image_g)
            
            image[i*tile_size:last_x, j*tile_size:last_y] = copy.deepcopy(image_chunk[:chunk_x,:chunk_y])

            #print('%3.2f percent done with rasterization' % ((100.0 * (i * ver_tiles + j)) / (hor_tiles * ver_tiles)), end='\r')
    if(training):
        cl.enqueue_copy(queue, d_colors, d_colors_g)
        cl.enqueue_copy(queue, d_centers, d_centers_g)
        cl.enqueue_copy(queue, d_conics, d_conics_g)
        cl.enqueue_copy(queue, d_opacity, d_opacity_g)
    
    return image, d_colors, d_centers, d_conics, d_opacity

def c_rasterize(centers: np.ndarray, colors: np.ndarray, opacities: np.ndarray, conics: np.ndarray, mapped_keys,
              training=True, result_size=[979,546], tile_size=16, background_color=np.zeros(3, dtype=np.float64)):

    FUNC = CDLL("./shared.so")
    FUNC.rasterize_tile.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'), #image
                                    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS'), #last_contributors
                                    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # centers
                                    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # colors
                                    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # opacities
                                    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # conics
                                    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'), # depth_sorted_gaus
                                    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS') # background_color
                                    ]
    image = np.empty([result_size[0], result_size[1], 3])
    image_chunk = np.zeros([tile_size, tile_size, 3])
    last_contributors = np.empty([result_size[0], result_size[1]], dtype=np.int32)
    last_contributors_chunk = np.empty([tile_size, tile_size], dtype=np.int32)

    hor_tiles = math.ceil(result_size[0] / tile_size)
    ver_tiles = math.ceil(result_size[1] / tile_size)

    for i in range(hor_tiles):
        for j in range(ver_tiles):
            depth_sorted = np.asarray(sorted(mapped_keys[i][j].items()), dtype=np.int32)
            if (len(depth_sorted > 0)):
                depth_sorted_gaus = np.ascontiguousarray(depth_sorted[...,1]) # flatten the dictionary to a depth-sorted 1D array
            else:
                depth_sorted_gaus = np.empty(0, dtype=np.int32)

            FUNC.rasterize_tile(image_chunk, last_contributors_chunk, centers, colors, opacities, conics,
                                depth_sorted_gaus, background_color, len(depth_sorted_gaus), training, i, j, result_size[0], result_size[1])
            
            last_x = min((i+1)*tile_size,result_size[0])
            last_y = min((j+1)*tile_size,result_size[1])
            chunk_x = min(tile_size, result_size[0]-i*tile_size)
            chunk_y = min(tile_size, result_size[1]-j*tile_size)
            image[i*tile_size:last_x, j*tile_size:last_y] = copy.deepcopy(image_chunk[:chunk_x,:chunk_y])
            last_contributors[i*tile_size:last_x, j*tile_size:last_y] = copy.deepcopy(last_contributors_chunk[:chunk_x,:chunk_y])

            image_chunk = np.empty([tile_size, tile_size, 3])
            last_contributors_chunk = np.empty([tile_size, tile_size], dtype=np.int32)

            print('%3.2f percent done with rasterization' % ((100.0 * (i * ver_tiles + j)) / (hor_tiles * ver_tiles)), end='\r')

    return image

def rasterize(centers, colors, opacities, conics, mapped_keys,
              training=True, result_size=[979,546], tile_size=16, background_color=np.zeros(3)):
    image = np.empty([result_size[0], result_size[1], 3])
    last_contributors = np.empty(result_size)

    hor_tiles = math.ceil(result_size[0] / tile_size)
    ver_tiles = math.ceil(result_size[1] / tile_size)

    contributor = 0

    for i in range(hor_tiles):
        for j in range(ver_tiles):
            depth_sorted_gaus = sorted(mapped_keys[i][j].items())

            tile_x = i * tile_size
            tile_y = j * tile_size
            for x in range(tile_size):
                curr_x = tile_x+x
                if(curr_x >= result_size[0]):
                        break
                
                for y in range(tile_size):
                    curr_y = tile_y+y
                    if(curr_y >= result_size[1]):
                        break

                    contributor = 0
                    T = 1.0
                    color = np.zeros(3)
                    for depth, gaus in depth_sorted_gaus:
                        contributor += 1

                        pix_dist_x = tile_x + x + .5
                        pix_dist_y = tile_y + y + .5

                        distance_x = centers[gaus,0] - pix_dist_x
                        distance_y = centers[gaus,1] - pix_dist_y

                        print('{a} {b}'.format(a=centers[gaus,0], b=centers[gaus,1]))
                        conic = conics[gaus]

                        power = -.5 * (conic[0] * distance_x * distance_x + conic[2] * distance_y * distance_y) - conic[1] * distance_x * distance_y

                        if (power > 0):
                            continue

                        alpha = min(.99, opacities[gaus] * math.exp(power))
                        if (alpha < (1.0/255)):
                            continue

                        test_T = T * (1 - alpha)
                        if (test_T < .0001):
                            break

                        color += colors[gaus] * alpha * T
                        T = test_T
                    last_contributors[curr_x,curr_y] = contributor
                    image[curr_x,curr_y] = color + background_color * T
            print('%3.2f percent done with rasterization' % ((100.0 * (i * ver_tiles + j)) / (hor_tiles * ver_tiles)), end='\r')

    return image