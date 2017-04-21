import tensorflow as tf
import data_batcher
import os
from collections import namedtuple

HParams = namedtuple('HParams', ['num_hidden_layers', 'hidden_units', 'batch_size', 'lr', 'min_lr', 'num_steps', 'max_grad_norm', 'epochs', 'dropout'])


#hyperparameters for neural network

hps = HParams(num_hidden_layers=0,
              hidden_units=512,
              batch_size=50,
              lr=0.5,
              min_lr=0.01,
              num_steps=1000,
              max_grad_norm=5.0,
              epochs=4,
              dropout=1.0)

class MNIST_classifier(object):

    def __init__(self, hps, gpu=None):
        self.hps = hps
        self.gpu = gpu

    def _add_placeholders(self):
        self.images = tf.placeholder(tf.float32, [self.hps.batch_size, 784], name="images")
        self.labels = tf.placeholder(tf.int64, [self.hps.batch_size], name="labels")  # labels
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def build_graph(self):
        if self.gpu == None:
            device = None
        elif self.gpu == False:
            device = '/cpu:0'
        elif self.gpu == True:
            device = '/gpu:0'
        tf.reset_default_graph()
        hps = self.hps
        self._add_placeholders()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        x = self.images
        y_ = self.labels

        initializer = tf.random_uniform_initializer(-.1, .1)

        with tf.device(self.gpu):
            if hps.num_hidden_layers == 0: #logistic regression
                W = tf.get_variable("weight_matrix", [784,10], initializer=initializer)
                b = tf.get_variable("bias", 10, initializer=initializer)
                with tf.variable_scope('logits'):
                    y = tf.matmul(x, W) + b
                    self.logits = y
                    self.predictions = tf.argmax(y, 1)
                tf.summary.histogram('weight_matrix', W)
                tf.summary.histogram('bias', b)

            elif hps.num_hidden_layers > 0:
                size_list = [784] + [hps.hidden_units for _ in range(hps.num_hidden_layers)]
                h = x #initialize with input layer
                for i in range(hps.num_hidden_layers):
                    with tf.variable_scope('hidden_layer_%d' % (i+1)):
                        Wh = tf.get_variable('W', [size_list[i], size_list[i+1]], initializer=initializer)
                        bh = tf.get_variable('b', hps.hidden_units, initializer=initializer)
                        h = tf.nn.sigmoid(tf.matmul(h,Wh) + bh, 'hidden_state')
                        tf.summary.histogram('W', Wh)
                        tf.summary.histogram('b', bh+1)
                with tf.variable_scope('output'):
                    W = tf.get_variable('W', [hps.hidden_units, 10], initializer=initializer)
                    b = tf.get_variable('b', 10, initializer=initializer)
                    y = tf.matmul(h, W) + b
                    tf.summary.histogram('W', W)
                    tf.summary.histogram('b', b)
                    self.logits = y
                    self.predictions = tf.argmax(y,1)

            self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y)) #batch-averaged cross entropy loss
            tf.summary.scalar('loss', self.loss)

            a = int(hps.epochs / 2)
            b = hps.epochs
            learning_rate = tf.train.piecewise_constant(self.global_step, [i * hps.num_steps for i in range(a, b+1)],
                                                        [hps.lr * (0.9) ** max(0, epoch - a) for epoch in range(a, b + 2)])
            tf.summary.scalar('learning_rate', learning_rate)

            tvars = tf.trainable_variables()
            clipped_grads = [tf.clip_by_value(g, -100, 100, name='clipped_gradient') for g in tf.gradients(self.loss, tvars)]
            grads, global_norm = tf.clip_by_global_norm(clipped_grads, hps.max_grad_norm)
            tf.summary.scalar('global_norm', global_norm)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.train_op = optimizer.apply_gradients(
                zip(grads, tvars), global_step=self.global_step, name='train_step')

            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
            self.accuracy_per_batch = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy_per_batch', self.accuracy_per_batch)
            self.merged_summary = tf.summary.merge_all()

def Train(model, save_path):
    hps = model.hps
    tf.reset_default_graph()
    model.build_graph()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter(save_path)
        writer.add_graph(sess.graph)
        batcher = data_batcher.Batcher('train')
        for epoch in range(hps.epochs):
            if epoch > 0:
                batcher.shuffle()
            for step in range(hps.num_steps):
                batch_images, batch_labels = batcher.next_batch(hps.batch_size)
                _, l, acc, summaries, gstep = sess.run([model.train_op, model.loss, model.accuracy_per_batch, model.merged_summary, model.global_step],
                                                       feed_dict={model.images: batch_images, model.labels: batch_labels, model.keep_prob:hps.dropout})
                if gstep % 20 == 0:
                    writer.add_summary(summaries, gstep) #records summaries to tensorboard file
        saver = tf.train.Saver() #for model restoration
        saver.save(sess, save_path + 'model.ckpt')
        writer.close()
        print('Training complete.')

        # Test trained model
        test_batcher = data_batcher.Batcher('test')
        N = 10000 // hps.batch_size
        accuracy = 0
        for _ in range(N):
            test_images, test_labels = test_batcher.next_batch(hps.batch_size)
            accuracy += sess.run(model.accuracy_per_batch, feed_dict={model.images: test_images, model.labels: test_labels, model.keep_prob:1.0})
        accuracy = accuracy/N
        print('Test Accuracy:', accuracy)
        with open(save_path + 'test_accuracy.txt', 'w') as f:
            f.write('Hyperparameters: ' + str(hps) + '\n\n')
            f.write('Test Accuracy: ' + str(accuracy))

if __name__ == '__main__':
    if not os.path.exists('./output'):
        os.makedirs('./output')
    with open('./output/hps.txt', 'w') as f:
        f.write(str(hps))  # save hyperparameters for restoring and testing model later
    model = MNIST_classifier(hps)
    Train(model, './output/')