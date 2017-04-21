import gzip
import numpy as np

class Batcher(object):

    def __init__(self, mode, shuffle=True):

        image_size = 28
        self.counter = 0 #start position of next batch

        if mode == 'train':

            self.num_images = 60000
            num_images = self.num_images

            with gzip.open('data/train-images-idx3-ubyte.gz') as bytestream:
                bytestream.read(16)

                buf = bytestream.read(image_size * image_size * num_images)
                data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
                data = data.reshape(num_images, image_size*image_size)
                self.images = [data[i,:] for i in range(num_images)]

            with gzip.open('data/train-labels-idx1-ubyte.gz') as bytestream:
                bytestream.read(8)
                buf = bytestream.read(1 * num_images)
                labels = list(np.frombuffer(buf, dtype=np.uint8).astype(np.int64))
                self.labels = labels

        elif mode == 'test' or mode == 'eval':

            self.num_images = 10000
            num_images = self.num_images

            with gzip.open('data/t10k-images-idx3-ubyte.gz') as bytestream:
                bytestream.read(16)
                buf = bytestream.read(image_size * image_size * num_images)
                data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
                data = data.reshape(num_images, image_size*image_size)
            self.images = [data[i, :] for i in range(num_images)]

            with gzip.open('data/t10k-labels-idx1-ubyte.gz') as bytestream:
                bytestream.read(8)
                buf = bytestream.read(1 * num_images)
                labels = list(np.frombuffer(buf, dtype=np.uint8).astype(np.int64))
            self.labels = labels

        if shuffle == True:
            self.shuffle()

    def next_batch(self, batch_size):
        images = [self.images[i] for i in range(self.counter, self.counter + batch_size)]
        labels = [self.labels[i] for i in range(self.counter, self.counter + batch_size)]
        self.counter = (self.counter + batch_size) % (self.num_images - batch_size)
        return images, labels

    def shuffle(self):
        perm = np.random.permutation(self.num_images)
        self.images = [self.images[i] for i in perm]
        self.labels = [self.labels[i] for i in perm]

def make_one_hot_vector(label): #label is an integer
    x = np.zeros(10)
    x[label] = 1
    return x

def make_one_hot(labels): #labels is a list of integers
    return [make_one_hot_vector(label) for label in labels]