
import tensorflow as tf
import tools

#%%
def VGG16(x, n_classes, is_pretrain=True):  #is_pretrain为False则固定住参数不改变
    
    x = tools.conv('conv1_1', x, out_channels=64, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv1_2', x, out_channels=64, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool1', x, kernel=[1,8,8,8,1], stride=[1,8,8,8,1], is_max_pool=True)
    
    x = tools.conv('conv2_1', x, out_channels=128, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv2_2', x, out_channels=128, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool2', x, kernel=[1,8,8,8,1], stride=[1,8,8,8,1], is_max_pool=True)
    
    x = tools.conv('conv3_1', x, out_channels=256, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv3_2', x, out_channels=256, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv3_3', x, out_channels=256, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool3', x, kernel=[1,8,8,8,1], stride=[1,8,8,8,1], is_max_pool=True)
    
    x = tools.conv('conv4_1', x, out_channels=512, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv4_2', x, out_channels=512, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv4_3', x, out_channels=512, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool3', x, kernel=[1,8,8,8,1], stride=[1,8,8,8,1], is_max_pool=True)

    x = tools.conv('conv5_1', x, out_channels=512, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv5_2', x, out_channels=512, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
    x = tools.conv('conv5_3', x, out_channels=512, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
    x = tools.pool('pool3', x, kernel=[1,8,8,8,1], stride=[1,8,8,8,1], is_max_pool=True)

    x = tools.FC_layer('fc6', x, out_nodes=1024)
    x = tools.batch_norm(x) # or dropout
    x = tools.FC_layer('fc7', x, out_nodes=1024)
    x = tools.batch_norm(x) # or dropout
    x = tools.FC_layer('fc8', x, out_nodes=n_classes)

    return x
        




#%% TO get better tensorboard figures!

def VGG16N(x, n_classes, is_pretrain=True): # True则可在训练时改变
    
    with tf.name_scope('VGG16'):

        x = tools.conv('conv1_1', x, out_channels=32, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv1_2', x, out_channels=32, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool1'):    
            x = tools.pool('pool1', x, kernel=[1,8,8,8,1], stride=[1,8,8,8,1], is_max_pool=True)
            
        x = tools.conv('conv2_1', x, out_channels=32, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv2_2', x, out_channels=32, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool2'):    
            x = tools.pool('pool2', x, kernel=[1,8,8,8,1], stride=[1,8,8,8,1], is_max_pool=True)

        x = tools.conv('conv3_1', x, out_channels=32, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv3_2', x, out_channels=32, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv3_3', x, out_channels=32, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool3'):
            x = tools.pool('pool3', x, kernel=[1,8,8,8,1], stride=[1,8,8,8,1], is_max_pool=True)

        x = tools.conv('conv4_1', x, out_channels=32, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv4_2', x, out_channels=32, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv4_3', x, out_channels=32, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool4'):
            x = tools.pool('pool4', x, kernel=[1,8,8,8,1], stride=[1,8,8,8,1], is_max_pool=True)

        x = tools.conv('conv5_1', x, out_channels=512, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv5_2', x, out_channels=512, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv5_3', x, out_channels=512, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool5'):
            x = tools.pool('pool5', x, kernel=[1,8,8,8,1], stride=[1,8,8,8,1], is_max_pool=True)

        x = tools.FC_layer('fc6', x, out_nodes=256)
        with tf.name_scope('batch_norm1'):
            x = tools.batch_norm(x)
        x = tools.FC_layer('fc7', x, out_nodes=256)
        with tf.name_scope('batch_norm2'):
            x = tools.batch_norm(x)
        x = tools.FC_layer('fc8', x, out_nodes=n_classes)
    
        return x

# 改动第一层以便迁移学习
def VGG16T(x, n_classes, is_pretrain=True): # True则可在训练时改变
    with tf.name_scope('VGG16'):
        x = tools.conv2D('conv1_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        x = tools.conv2D('conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=False)
        with tf.name_scope('pool1'):
            x = tools.pool2D('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv2D('conv2_1', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=False)
        x = tools.conv2D('conv2_2', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=False)
        with tf.name_scope('pool2'):
            x = tools.pool2D('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv2D('conv3_1', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=False)
        x = tools.conv2D('conv3_2', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=False)
        x = tools.conv2D('conv3_3', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=False)
        with tf.name_scope('pool3'):
            x = tools.pool2D('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv2D('conv4_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=False)
        x = tools.conv2D('conv4_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=False)
        x = tools.conv2D('conv4_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=False)
        with tf.name_scope('pool4'):
            x = tools.pool2D('pool4', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.conv2D('conv5_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=False)
        x = tools.conv2D('conv5_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=False)
        x = tools.conv2D('conv5_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_pretrain=False)
        with tf.name_scope('pool5'):
            x = tools.pool2D('pool5', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = tools.FC_layer('fc6', x, out_nodes=4096)
        with tf.name_scope('batch_norm1'):
            x = tools.batch_norm(x)
        x = tools.FC_layer('fc7', x, out_nodes=4096)
        with tf.name_scope('batch_norm2'):
            x = tools.batch_norm(x)
        x = tools.FC_layer('fc8', x, out_nodes=n_classes)

        return x

#%%







            
