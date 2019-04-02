# -*- coding: utf-8 -*-
#!/usr/bin/env python
######################################
####    MDD_sMRI_classification   ####
######################################

#### Version    : 1.0
#### Date       : Mar. 2019 ~ Sep. 2018
#### Author     : Freeman Fu

#%%
#DATA:
    #1. sMRI data
    #2. pratrained weights (vgg16.npy):https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
    
# TO Train and test:
    #0. get data ready, get paths ready !!!
    #1. run training_and_val.py and call train() in the console
    #2. call evaluate() in the console to test
    
#%%
import os
import os.path
import numpy as np
import tensorflow as tf
import math
import input_data
import VGG
import tools
#%%
IMG_W = 121
IMG_H = 145
IMG_D = 121
N_CLASSES = 2
BATCH_SIZE = 4
learning_rate = 0.00001
MAX_STEP = 15000   #
IS_PRETRAIN = False

#%%   Training
def train():

    pre_trained_weights = 'E:\python_project\cats_vs_dogs\My-TensorFlow-tutorials-master/04_VGG_Tensorflow\VGG16_pretrained/vgg16.npy'    #加载训练过的模型参数
    data_dir = 'E:\\python_project\\MY_FMRI_CNN\\data\\train\\'
    train_log_dir = 'E:\\python_project\\MY_MRI_VGG\\logs\\train'   #
    val_log_dir = 'E:\\python_project\\MY_MRI_VGG\\logs\\validation'#分别存两个日志方便以后画两个曲线图
    
    with tf.name_scope('input'):
        tra_image, tra_label, val_image, val_label = input_data.get_files(data_dir,
                                                                         ratio=0.2)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W * IMG_H * IMG_D])  #
    x_3d = tf.reshape(x, [-1, IMG_W, IMG_H, IMG_D, 1])  # [-1,121,145, 121, 1]
    x_2d = tf.reshape(x, [-1, IMG_W, IMG_H, IMG_D])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    
    logits = VGG.VGG16T(x_2d, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False) 
    train_op = tools.optimize(loss, learning_rate, my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()  #

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    # load the parameter file, assign the parameters, skip the specific layers
    #有选择地加载训练过的模型参数
    tools.load_with_skip(pre_trained_weights, sess, ['conv1_1','fc6','fc7','fc8'])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            tra_image_batch, tra_label_batch = input_data.get_batch(tra_image,
                                                                    tra_label,
                                                                   BATCH_SIZE,
                                                                  )
            #tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                       feed_dict={x:tra_image_batch, y_:tra_label_batch})
            if step % 50 == 0 or (step + 1) == MAX_STEP:                 
                print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op,
                                       feed_dict={x:tra_image_batch, y_:tra_label_batch})
                tra_summary_writer.add_summary(summary_str, step)
                
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                val_image_batch, val_label_batch = input_data.get_val_batch(val_image,
                                                                         val_label,
                                                                         BATCH_SIZE,
                                                                         )
                #val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                        feed_dict={x:val_image_batch,y_:val_label_batch} )
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc*100.0))

                summary_str = sess.run(summary_op,
                                       feed_dict={x:val_image_batch,y_:val_label_batch})
                val_summary_writer.add_summary(summary_str, step)
                    
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()





    
#%%   Test the accuracy on test dataset. got about 85.69% accuracy.

def evaluate():
    # with tf.Graph().as_default():
    #     log_dir = 'E:\\python_project\\MY_MRI_VGG\\logs\\train\\'
    #     test_dir = 'E:\\python_project\\MY_FMRI_CNN\\data\\train\\'
    #
    #     test_images, test_labels, _, _ = input_data.get_files(test_dir,ratio=0)
    #     n_test = len(test_images)
    #     BATCH_SIZE = 1
    #
    #     x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W * IMG_H * IMG_D])  #
    #     x_ = tf.reshape(x, [-1, IMG_W, IMG_H, IMG_D, 1])  # [-1,121,145, 121, 1]
    #     y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    #
    #     logits = VGG.VGG16T(x_, N_CLASSES, IS_PRETRAIN)
    #     correct = tools.num_correct_prediction(logits, y_)
    #     testloss = tools.loss(logits, y_)
    #     saver = tf.train.Saver(tf.global_variables())
    #
    #     with tf.Session() as sess:
    #         print("Reading checkpoints...")
    #         ckpt = tf.train.get_checkpoint_state(log_dir)
    #         if ckpt and ckpt.model_checkpoint_path:
    #             global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    #             saver.restore(sess, ckpt.model_checkpoint_path)
    #             print('Loading success, global_step is %s' % global_step)
    #         else:
    #             print('No checkpoint file found')
    #             return
    #
    #         coord = tf.train.Coordinator()
    #         threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    #
    #         try:
    #             print('\n......Evaluating......')
    #             num_step = int(math.floor(n_test / BATCH_SIZE))
    #             num_sample = num_step*BATCH_SIZE
    #             step = 0
    #             total_correct = 0
    #             total_loss = 0
    #             sensitivity = []  # aka recall
    #             specificity = []  # aka True Nagative Rate
    #             test_y  = []
    #             while step < num_step and not coord.should_stop():
    #                 test_batch, test_label_batch = input_data.get_test_batch(test_images,
    #                                                                          test_labels,
    #                                                                          BATCH_SIZE,
    #                                                                          )
    #                 batch_correct = sess.run(correct,
    #                                     feed_dict={x: test_batch, y_: test_label_batch})
    #                 total_correct += np.sum(batch_correct)
    #                 batch_loss = sess.run([testloss],
    #                                     feed_dict={x: test_batch, y_: test_label_batch})
    #                 total_loss += np.sum(batch_loss)
    #                 if test_label_batch == 1:
    #                     sensitivity.append(batch_correct)
    #                 if test_label_batch == 0:
    #                     specificity.append(batch_correct)
    #                 test_y.append(int(test_label_batch))
    #                 step += 1
    #             sens = sum(sensitivity) / len(sensitivity)
    #             spec = sum(specificity) / len(specificity)
    #
    #             print('Total testing samples: %d' %num_sample)
    #             print('Total correct predictions: %d' %total_correct)
    #             print('The model\'s loss is %.2f' % (total_loss / num_step))
    #             print('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
    #             print('The sensitivity in test images are %.2f%%' % (sens * 100))
    #             print('The specificity in test images are %.2f%%' % (spec * 100))
    #             print(test_y)
    #         except Exception as e:
    #             coord.request_stop(e)
    #         finally:
    #             coord.request_stop()
    #             coord.join(threads)
    test_dir = 'E:\python_project\MY_FMRI_CNN/data/test/'
    N_CLASSES = 2
    BATCH_SIZE = 1
    print('-------------------------')
    test, test_label,_,__ = input_data.get_files(test_dir,ratio=0)
    n_test = len(test)
    print('There are %d test images totally..' % n_test)
    print('-------------------------')

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W*IMG_H*IMG_D])  #
    x_3d = tf.reshape(x, [-1, IMG_W, IMG_H, IMG_D, 1])  # [-1,121,145, 121, 1]
    x_2d = tf.reshape(x, [-1, IMG_W, IMG_H, IMG_D])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    logits = VGG.VGG16T(x_2d, N_CLASSES, IS_PRETRAIN)
    testloss = tools.loss(logits, y_)
    testacc = tools.accuracy(logits, y_)

    logs_train_dir = 'E:\\python_project\\MY_MRI_VGG\\logs\\train\\'
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
            return

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('------------Evaluating-------------')
        num_step = int(math.floor(n_test / BATCH_SIZE))
        real_test_num = num_step * BATCH_SIZE
        step = 0
        total_loss = 0
        total_acc = 0
        sensitivity = []  # aka recall
        specificity = []  # aka True Nagative Rate
        test_label_batch_ = []
        while step < num_step and not coord.should_stop():

            test_batch, test_label_batch = input_data.get_test_batch(test,
                                                                     test_label,
                                                                     BATCH_SIZE,
                                                                     )
            test_label_batch_.append(int(test_label_batch))
            batch_loss, batch_acc = sess.run([testloss, testacc],
                                           feed_dict={x: test_batch, y_: test_label_batch})
            total_loss += np.sum(batch_loss)
            total_acc += np.sum(batch_acc)
            if test_label_batch == 1:
                sensitivity.append(batch_acc)
            if test_label_batch == 0:
                specificity.append(batch_acc)
            step += 1
        sens = sum(sensitivity) / len(sensitivity)
        spec = sum(specificity) / len(specificity)

        # f = open(txt_path, 'a')
        # f.write('\r\n' + 'iteration:' + str(i) + '\t')
        # f.write('accuracy:' + str(test_ACC) + '\t')
        # f.write('sensitivity:' + str(Sens) + '\t')
        # f.write('specificity:' + str(Spec))
        # f.close()

        print('Total testing samples: %d' % real_test_num)
        print('The model\'s loss is %.2f' % (total_loss/num_step))
        num_correct = int(real_test_num*total_acc/num_step)
        print('Correct : %d' % num_correct)
        print('Wrong : %d' % (real_test_num - num_correct))
        print('The accuracy in test images are %.2f%%' %(total_acc*100.0/num_step))
        print('The sensitivity in test images are %.2f%%' %(sens*100))
        print('The specificity in test images are %.2f%%' %(spec*100))
    coord.request_stop()
    coord.join(threads)
    sess.close()
                
#%%
import nibabel as nib
def evaluate_one():
    '''Test one image against the saved models and parameters
    '''
    # you need to change the directories to yours.
    train_dir = 'E:\python_project\MY_FMRI_CNN\data/fff/sm0wrp1s20090306_07_ZangYF_LSF_JiaYue-0003-00001-000128-01.nii'

    FA_org = nib.load(train_dir)
    image_array = np.array(FA_org.get_data())  # 121x145x121; numpy.ndarray
    image_array = image_array.reshape(1, IMG_W, IMG_H, IMG_D, 1)

    with tf.Graph().as_default():
        N_CLASSES = 2

        image = tf.cast(image_array, tf.float32)

        logit = VGG.VGG16T(image, N_CLASSES, IS_PRETRAIN)

        logit = tf.nn.softmax(logit)  # 激活函数

        x = tf.placeholder(tf.float32, shape=[1, IMG_W, IMG_H, IMG_D, 1])

        # you need to change the directories to yours.
        logs_train_dir = 'E:\\python_project\\MY_MRI_VGG\\logs\\train\\'

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('This is a health with possibility %.6f' % prediction[:, 0])
            else:
                print('This is a MDD with possibility %.6f' % prediction[:, 1])

#train()
evaluate()
#evaluate_one()