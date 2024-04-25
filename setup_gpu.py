import pyopencl as cl

def setup_gpu():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    program = cl.Program(ctx, """
    __kernel void rasterize(
        __global const double *centers, __global const double *colors, __global const double *opacities, __global const double *conics,
        __global const int *gaus_stack, __global const double *background_color, __global const int *other_data,
        __global double *image_chunk, __global int *last_contributors)
    {
        int gid = get_global_id(0);
        int loc_x = gid / 16;
        int loc_y = gid % 16;
        int x = other_data[2] + loc_x;
        int y = other_data[3] + loc_y;

        if(x >= other_data[4] || y >= other_data[5])
            return;
        
        int contributor = 0;
        int gaus;

        double T = 1.0;
        double dist_x, dist_y, power, alpha, test_T;

        for(int ch = 0; ch < 3; ch++) {
            image_chunk[gid*3+ch] = 0.0;
        }

        for (int i = 0; i < other_data[0]; i++) {
            contributor += 1;
            gaus = gaus_stack[i];

            dist_x = centers[gaus*2] - x + .5;
            dist_y = centers[gaus*2+1] - y + .5;

            power = -.5 * (conics[gaus*3] * dist_x * dist_x + conics[gaus*3+2] * dist_y * dist_y) - conics[gaus*3+1] * dist_x * dist_y;

            if (power > 0)
                continue;

            alpha = fmin(.99, opacities[gaus] * exp(power));
            if (alpha < (1.0/255))
                continue;

            test_T = T * (1.0 - alpha);
            if (test_T < .0001)
                break;

            for(int ch = 0; ch < 3; ch++) {
                image_chunk[gid*3+ch] += colors[gaus*3+ch] * alpha * T;
            }
            T = test_T;
        }

        for(int ch = 0; ch < 3; ch++) {
            image_chunk[gid*3+ch] += background_color[ch] * T;
        }
        last_contributors[gid] = contributor;

    }       
    """).build()

    return ctx, queue, program