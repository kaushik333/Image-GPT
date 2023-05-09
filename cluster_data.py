import numpy as np
from sklearn.cluster import MiniBatchKMeans
import argparse
import torchvision

def main():
    
    train_dataset = torchvision.datasets.CIFAR10(root="./cifar10_data", train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10(root="./cifar10_data", train=False, download=True, transform=torchvision.transforms.ToTensor())

    train_x = np.asarray([np.asarray(x[0]) for x in train_dataset])
    test_x = np.asarray([np.asarray(x[0]) for x in test_dataset])
    print(train_x.shape, test_x.shape)

    train_x = train_x.reshape(-1, train_x.shape[1])
    print(train_x.shape)

    kmeans = MiniBatchKMeans(n_clusters=16, random_state=0, batch_size=1024, verbose=1).fit(train_x)
    print(kmeans.cluster_centers_.shape)

    np.save("./cifar10_data/cifar10_centroids_16.npy", kmeans.cluster_centers_)


if __name__ == "__main__":
    main()