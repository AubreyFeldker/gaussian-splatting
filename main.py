from train import train_model
from colmap import read_write_model as colmap_rw
import multiprocessing, time

if __name__ == "__main__":
    path = "C:/Users/Brooks/Downloads/colmap_db/truck/sparse/0"
    cameras, images, point_cloud = colmap_rw.read_model(path, ".bin")

    learning_rates = {
        'center': .00016,
        'scaling': .005,
        'opacity': .05,
        'rotation': .001
    }

    print(multiprocessing.cpu_count())
    t1 = time.perf_counter()
    train_model(cameras, images, point_cloud, learning_rates)
    print()
    print(time.perf_counter() - t1)