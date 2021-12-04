import tensorflow as tf
resnet=tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
#Create model
#Add resnet backbone

finetuning = True

if finetuning:
    freeze_until = 15 # layer from which we want to fine-tune
    
    for layer in resnet.layers[:freeze_until]:
        layer.trainable = False
else:
    resnet.trainable = False



conv6=tf.keras.layers.Conv2D(filters=1024,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same')(resnet.get_layer("post_relu").output)
relu1=tf.keras.layers.ReLU()(conv6)
#Upsampling block 1
conc1=tf.keras.layers.concatenate([relu1,resnet.get_layer("conv4_block36_out")])

conv7=tf.keras.layers.Conv2D(filters=512,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same')(conc1)
relu2=tf.keras.layers.ReLU()(conv7)
up1=tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(relu2)

#Upsampling block 2
conc2=tf.keras.layers.Concatenate([up1,resnet.get_layer("conv3_block8_out")],-1)

conv8=tf.keras.layers.Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same')(conc2)
relu3=tf.keras.layers.ReLU()(conv8)
up2=tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(relu3)

#Upsampling block 3
conc3=tf.keras.layers.Concatenate([up2,resnet.get_layer("conv2_block3_out")],-1)

conv9=tf.keras.layers.Conv2D(filters=128,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same')(conc3)
relu4=tf.keras.layers.ReLU()(conv9)
up3=tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(relu4)

#Upsampling block 4
conc4=tf.keras.layers.Concatenate([up3,resnet.get_layer("pool1_pool")],-1)

conv10=tf.keras.layers.Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same')(conc4)
relu5=tf.keras.layers.ReLU()(conv10)
up4=tf.keras.layers.UpSampling2D(4, interpolation='bilinear')(relu5)

#Mix and predict
conv11=tf.keras.layers.Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same')(up4)
relu6=tf.keras.layers.ReLU()(conv11)
output=tf.keras.layers.Conv2D(filters=2,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='softmax')(relu6)

model=tf.keras.Model(resnet.input,output)

