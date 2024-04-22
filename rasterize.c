#include <stdio.h>
#include <windows.h>

const int TILE_SIZE = 16;

float rasterize_tile(double image_chunk[TILE_SIZE][TILE_SIZE][3], int last_contributors[TILE_SIZE][TILE_SIZE], double centers[][2], double colors[][3], double opacities[], float conics[][3],
                   int gaus_stack[], double background_color[3], int stack_size, boolean training, int result_x, int result_y) {
    printf("%d\n", stack_size);
    printf("Centers before setting: %d\n", centers[1][0]);
    centers[0][0] = .12;
    
    return centers[0][0];
    // TODO: implement the c based tile rasterizer
}