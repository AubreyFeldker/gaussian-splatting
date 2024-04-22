import numpy as np, math

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




