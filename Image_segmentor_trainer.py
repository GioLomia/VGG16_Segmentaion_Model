import LoadBatches
import VGGSegnet
from keras.optimizers import Nadam

train_images_path = "C:/Users/lomiag/Desktop/Object Transfer Project/Data/dataset1/dataset1/images_prepped_train/"
train_segs_path = "C:/Users/lomiag/Desktop/Object Transfer Project/Data/dataset1/dataset1/annotations_prepped_train/"
train_batch_size = "C:/Users/lomiag/Desktop/Object Transfer Project/Data/dataset1/dataset1/images_prepped_test/"
val_images = "C:/Users/lomiag/Desktop/Object Transfer Project/Data/dataset1/dataset1/images_prepped_test/"
val_annotations = "C:/Users/lomiag/Desktop/Object Transfer Project/Data/dataset1/dataset1/annotations_prepped_test/"
n_classes = 10
input_height = 224
input_width = 224
validate = True
save_weights_path = "C:/Users/lomiag/Desktop/Object Transfer Project/Weights/Weights"
epochs = 5
load_weights = 0


optimizer_name = Nadam
model_name = "vgg_segnet"

if validate:
    val_images_path = "C:/Users/lomiag/Desktop/Object Transfer Project/Data/dataset1/dataset1/images_prepped_test/"
    val_segs_path = "C:/Users/lomiag/Desktop/Object Transfer Project/Data/dataset1/dataset1/annotations_prepped_test/"
    val_batch_size = 20

modelFns = {'vgg_segnet': VGGSegnet.VGGSegnet}
modelFN = modelFns['vgg_segnet']

m = modelFN(n_classes, input_height=input_height, input_width=input_width)
m.compile(loss='categorical_crossentropy',
          optimizer="Adam",
          metrics=['accuracy'])

#
# if len( load_weights ) > 0:
# 	m.load_weights(load_weights)


print("Model output shape", m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth

G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                           input_height, input_width, output_height, output_width)

if validate:
    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, input_height,
                                                input_width, output_height, output_width)

if not validate:
    for ep in range(epochs):
        m.fit_generator(G, 512, epochs=1)
        m.save_weights(save_weights_path + "." + str(ep))
        m.save(save_weights_path + ".model." + str(ep))
else:
    for ep in range(epochs):
        m.fit_generator(G, 512, validation_data=G2, validation_steps=200, epochs=1)
        m.save_weights(save_weights_path + "." + str(ep))
        m.save(save_weights_path + ".model." + str(ep))
#
