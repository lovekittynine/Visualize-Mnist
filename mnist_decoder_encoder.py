#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:56:50 2018

@author: wsw
"""

# encoder and decoder
import tensorflow as tf
import numpy as np
from keras.layers import UpSampling2D
from timer import Timer
slim = tf.contrib.slim

tf.reset_default_graph()

def build_model(inputs,is_training=True):
    
    init = tf.contrib.layers.xavier_initializer_conv2d()
    batchnorm_param = {'decay':0.9,
                       'updates_collections':None,
                       'zero_debias_moving_mean':True,
                       'is_training':is_training}
    
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=init,
                        weights_regularizer=slim.l2_regularizer(5e-4),
                        biases_initializer=None,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batchnorm_param,
                        activation_fn=tf.nn.relu,
                        kernel_size=[3,3],
                        padding='SAME'):
        # encoder
        conv1 = slim.conv2d(inputs,num_outputs=32,stride=1,scope='conv1')
        pool1 = slim.max_pool2d(conv1,kernel_size=[2,2],stride=2,scope='pool1')
        conv2 = slim.conv2d(pool1,num_outputs=64,stride=1,scope='conv2')
        pool2 = slim.max_pool2d(conv2,kernel_size=[2,2],stride=2,scope='pool2')
        global_avg_pool = slim.max_pool2d(pool2,kernel_size=[7,7],stride=1,
                                          scope='global_avg_pool1')
        flat1 = slim.flatten(global_avg_pool,scope='flatten1')
        logits = slim.fully_connected(flat1,num_outputs=10,
                                      activation_fn=None,scope='fc1')
        
        if is_training:
            # decoder
            w1 = slim.get_variables_by_name('conv1/weights')[0]
            w2 = slim.get_variables_by_name('conv2/weights')[0]
            # unpool1
            # unpool1 = UpSampling2D(size=(2,2))(pool1)
            
            # conv1 transpose
            trans_conv1 = tf.nn.conv2d_transpose(pool1,
                                                 filter=w1,
                                                 output_shape=tf.shape(inputs),
                                                 strides=[1,2,2,1],
                                                 padding='SAME',
                                                 name='conv2d_trans1')
            # unpool2_1
            # unpool2_1 = UpSampling2D(size=(2,2))(pool2)
            
            # conv2_1 transpose
            trans_conv2_1 = tf.nn.conv2d_transpose(pool2,
                                                   filter=w2,
                                                   output_shape=tf.shape(pool1),
                                                   strides=[1,2,2,1],
                                                   padding='SAME',
                                                   name='conv2d_trans2')
            # unpool2_2
            # unpool2_2 = UpSampling2D()(trans_conv2_1)
            
            # conv2_2 transpose
            trans_conv2_2 = tf.nn.conv2d_transpose(trans_conv2_1,
                                                   filter=w1,
                                                   output_shape=tf.shape(inputs),
                                                   strides=[1,2,2,1],
                                                   padding='SAME',
                                                   name='conv2d_trans3')
            
            images = tf.concat([inputs,trans_conv1,trans_conv2_2],axis=2)
            tf.summary.image('images',images,max_outputs=5)
        
        if is_training:
            print('Unpool1 shape',pool1.get_shape().as_list())
            print('Unpool2 shape',trans_conv2_1.get_shape().as_list())
            print('Conv2_Feature shape',trans_conv2_2.get_shape().as_list())
            return logits,images
        
        else:
            return logits
        
        


def train():
    # normalize to 0-1
    train_data = np.load('./mnist/train.npy').astype(np.float32)/255.0
    train_label = np.load('./mnist/train-label.npy').astype(np.int32)
    
    test_data = np.load('./mnist/test.npy').astype(np.float32)/255.0
    test_data = test_data.reshape([-1,28,28,1])
    test_label = np.load('./mnist/test-label.npy').astype(np.int32)
    
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32,shape=[None,28,28,1])
        # if one demension must careful not use [None,1]
        ys = tf.placeholder(tf.int64,shape=[None,])
    
    with tf.name_scope('get_batch'):
        train_img,train_label = tf.train.slice_input_producer([train_data,train_label],
                                                               num_epochs=10)
        train_img = tf.reshape(train_img,shape=[28,28,1])
        # standarlization
        # train_img = tf.image.per_image_standardization(train_img)
        train_img_batch,train_lab_batch = tf.train.batch([train_img,train_label],
                                                          batch_size=128,
                                                          allow_smaller_final_batch=True)
    
    with tf.variable_scope('inference'):
        train_logits,images = build_model(xs,is_training=True)
        tf.get_variable_scope().reuse_variables()
        test_logits = build_model(xs,is_training=False)
    
    with tf.name_scope('loss'):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=ys,logits=train_logits)
        tf.summary.scalar('loss',loss)
    
    with tf.name_scope('compute_acc'):
        predict = tf.argmax(train_logits,axis=-1)
        evaluate = tf.equal(predict,ys)
        accuracy = tf.reduce_mean(tf.cast(evaluate,dtype=tf.float32))
        
    with tf.name_scope('train'):
        
        global_step = slim.train.create_global_step()
        ema = tf.train.ExponentialMovingAverage(0.999)
        # using average moving for loss
        average_op = ema.apply([loss])
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op.append(average_op)):
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss,global_step)
        
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    merge_op = tf.summary.merge_all()
    
    with tf.Session(config=config) as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        train_timer = Timer()
        writer = tf.summary.FileWriter('./logs',sess.graph,flush_secs=30)
        try:
            while not coord.should_stop():
                train_timer.tic()
                train_imgs,train_labs = sess.run([train_img_batch,train_lab_batch])
                loss_value,acc,_,summaries= sess.run([loss,accuracy,train_op,merge_op],
                                                      feed_dict={xs:train_imgs,ys:train_labs})
                train_timer.toc()
                step = global_step.eval()
                writer.add_summary(summaries,step)
                r = '\r>>>Step:{:5d} Loss:{:5.2f} Batch accu:{:.3f} Time:{:.3f}s/step'.format(step,
                               loss_value,acc,train_timer.average_time)
                print(r,end='',flush=True)
                if step%470==0:
                    compute_acc(sess,xs,test_logits,test_data,test_label)
        except tf.errors.OutOfRangeError:
            coord.request_stop()
            compute_acc(sess,xs,test_logits,test_data,test_label)
        finally:
            coord.join(threads)


def compute_acc(sess,xs,test_logits,imgs,labels):
    nums = len(imgs)
    accu_list = []
    print('\nComputing Test Accuracy...')
    for i in range(nums//100):
        start = i*100
        end = min(start+100,nums)
        logits = sess.run(test_logits,feed_dict={xs:imgs[start:end]})
        predicts = np.argmax(logits,axis=-1)
        accu = np.mean(np.equal(predicts,labels[start:end]))
        accu_list.append(accu)
        print('\r>>>Step:{:4d}/{:4d}'.format(i+1,nums//100),end='',flush=True)
    accuracy = np.mean(accu_list)
    print('\nTest accu:{:.3f}'.format(accuracy))


if __name__ == '__main__':
    train()
