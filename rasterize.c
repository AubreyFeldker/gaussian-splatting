#include <stdio.h>
#include <windows.h>
#include <math.h>

const int TILE_SIZE = 16;
const int CHANNELS = 3;

typedef struct thread_data {
    int pix_y;
} MYDATA, *PMYDATA;

int p_start_x;
int p_start_y;
int p_columns;
double (*p_image_chunk)[16][3];
int (*p_last_contributors)[16];
double (*p_centers)[2];
double (*p_colors)[3];
double *p_opacities;
float (*p_conics)[3];
int *p_gaus_stack;
double *p_background_color;
int p_stack_size;

DWORD WINAPI thread_rasterize(LPVOID param);

int rasterize_tile(double image_chunk[TILE_SIZE][TILE_SIZE][3], int last_contributors[TILE_SIZE][TILE_SIZE], double centers[][2], double colors[][3], double opacities[], float conics[][3],
                   int gaus_stack[], double background_color[3], int stack_size, boolean training, int tile_x, int tile_y, int result_x, int result_y) {
    DWORD dw_thread_id_array[TILE_SIZE];
    HANDLE h_thread_array[TILE_SIZE];

    // TODO: is there a more elegant way of doing this?
    // Putting pointers for all of the thread variables into global space
    p_start_x = tile_x * TILE_SIZE;
    p_start_y = tile_y * TILE_SIZE;
    p_columns = ((tile_x + 1) * TILE_SIZE > result_x) ? (result_x - tile_x * TILE_SIZE) : TILE_SIZE;
    p_image_chunk = image_chunk;
    p_last_contributors = last_contributors;
    p_centers = centers;
    p_colors = colors;
    p_opacities = opacities;
    p_conics = conics;
    p_gaus_stack = gaus_stack;
    p_background_color = background_color;
    p_stack_size = stack_size;

    int rows = ((tile_y + 1) * TILE_SIZE > result_y) ? (result_y - tile_y * TILE_SIZE) : TILE_SIZE;
    int row_num[TILE_SIZE]; // Set to an array cause otherwise pointer would change each iteration

    for (int i = 0; i < rows; i++) {
        //#region OLDTYPE
        /*p_data_array[i] = (PMYDATA) HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(MYDATA));

        p_data_array[i]->pix_x = tile_x * TILE_SIZE;
        p_data_array[i]->pix_y = tile_y * TILE_SIZE;
        p_data_array[i]->row_length = rows;
        p_data_array[i]->image_chunk = image_chunk;
        p_data_array[i]->last_contributors = last_contributors;
        p_data_array[i]->centers = centers;
        p_data_array[i]->colors = colors;*/
        //#endregion

        row_num[i] = i;

        h_thread_array[i] = CreateThread(
            NULL,
            0,
            thread_rasterize,
            &(row_num[i]),
            0,
            &dw_thread_id_array[i]);
    }

    WaitForMultipleObjects(rows, h_thread_array, TRUE, INFINITE);

    for(int i = 0; i < rows; i++)
        CloseHandle(h_thread_array[i]);

    return 0;
}

DWORD WINAPI thread_rasterize(LPVOID param) {
    int loc_y = *((int*)param);
    int y = loc_y + p_start_y;
    int x, contributor, gaus;
    double T, test_T, dist_x, dist_y, power, alpha;
    double *color;

    for (int loc_x = 0; loc_x < p_columns; loc_x++) {
        x = loc_x + p_start_x;
        contributor = 0;
        T = 1.0;
        color = p_image_chunk[loc_x][loc_y];

        for(int ch = 0; ch < CHANNELS; ch++) {
                p_image_chunk[loc_x][loc_y][ch] = 0.0;
            }

        for (int i = 0; i < p_stack_size; i++) {
            contributor++;
            gaus = p_gaus_stack[i];

            dist_x = p_centers[gaus][0] - ((double) x) + .5;
            dist_y = p_centers[gaus][1] - ((double) y) + .5;

            //printf("%f %f\n", p_centers[gaus][0], p_centers[gaus][1]);

            power = -.5 * (p_conics[gaus][0] * dist_x * dist_x + p_conics[gaus][2] * dist_y * dist_y) - p_conics[gaus][1] * dist_x * dist_y;
            if (power > 0)
                continue;

            alpha = fmin(.99, p_opacities[gaus] * exp(power));
            
            if (alpha < (1.0/255))
                continue;

            test_T = T * (1.0 - alpha);
            if (test_T < .0001)
                break;

            for(int ch = 0; ch < CHANNELS; ch++) {
                p_image_chunk[loc_x][loc_y][ch] += p_colors[gaus][ch] * alpha * T;
            }
            T = test_T;
        }
        
        for(int ch = 0; ch < CHANNELS; ch++) {
                p_image_chunk[loc_x][loc_y][ch] = p_image_chunk[loc_x][loc_y][ch] + p_background_color[ch] * T;
            }
        p_last_contributors[loc_x][loc_y] = contributor;
    }
}