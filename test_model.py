import tensorflow as tf
import data_batcher
import tensorflow_MNIST as MNIST
import matplotlib.pyplot as plt
from collections import namedtuple


HParams = namedtuple('HParams', ['num_hidden_layers', 'hidden_units', 'batch_size', 'lr', 'min_lr', 'num_steps', 'max_grad_norm', 'epochs', 'dropout'])

restore_file = './output/model.ckpt'

with open('./output/hps.txt', 'r') as f:
    hps = eval(f.read())   #reads in hyperparameters of model to be restored

model = MNIST.MNIST_classifier(hps)
model.build_graph()
sess = tf.Session()
tf.train.Saver().restore(sess, restore_file) #restores model
batcher = data_batcher.Batcher('test')
test_images, test_labels = batcher.next_batch(hps.batch_size)
predictions = sess.run(model.predictions, feed_dict={model.images:test_images, model.labels:test_labels})

for i in range(hps.batch_size): #sequentially display a batch size number of images along with the model's prediction
    if False: #if True, skip to the first image missclassified
        if predictions[i] == test_labels[i]:
            continue
    plt.title(('Image #%d is the letter ' % (i+1)) + str(predictions[i]))
    plt.imshow(test_images[i].reshape(28,28), cmap='gray')
    plt.show()
