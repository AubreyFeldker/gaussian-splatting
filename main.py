from train import train_model
from setup_gpu import setup_gpu
from colmap import read_write_model as colmap_rw
import pyopencl as cl, time, os

if __name__ == "__main__":
    t0 = time.perf_counter()
    path = "./input/sparse/0"
    cameras, images, point_cloud = colmap_rw.read_model(path, ".bin")

    learning_rates = {
        'center': .0008,
        'scaling': .025,
        'opacity': .25,
        'rotation': .005,
        'sh': .0125
    }

    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
    os.environ['PYOPENCL_CTX'] = '0'
    ctx, queue, program = setup_gpu()

    train_model(cameras, images, point_cloud, learning_rates, ctx, queue, program, t0=t0)