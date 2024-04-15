from alt_train import train
from colmap import read_write_model as colmap_rw

if __name__ == "__main__":
    path = "C:/Users/Brooks/Downloads/colmap_db/truck/sparse/0"
    cameras, images, point_cloud = colmap_rw.read_model(path, ".bin")

    learning_rates = {
        'center': .00016,
        'scaling': .005,
        'opacity': .05,
        'rotation': .001
    }

    #print()
    train(cameras, images, point_cloud, learning_rates)