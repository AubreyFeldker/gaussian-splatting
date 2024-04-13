from train import train
from colmap import read_write_model as colmap_rw

if __name__ == "__main__":
    path = "C:/Users/Brooks/Downloads/colmap_db/truck/sparse/0"
    cameras, images, point_cloud = colmap_rw.read_model(path, ".bin")

    #print(point_cloud)
    train(point_cloud)