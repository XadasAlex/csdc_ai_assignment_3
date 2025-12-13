# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.optimizers.schedules import ExponentialDecay
from keras import callbacks
from tensorflow.keras.optimizers import SGD, Adam
#from keras.utils import np_utils
from keras.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
from datetime import datetime
import numpy as np
import os

def display_classification_report(classification_report, figure_path, figure_name, onscreen=True):
    f = open(os.path.join(figure_path, figure_name+'.txt'), 'w')
    f.write(classification_report)
    f.close()

    if onscreen:
       print(classification_report)

def display_confusion_matrix(confusion_matrix, labels, figure_path,figure_name,figure_format,onscreen=True):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap=plt.cm.Greys)

    plt.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)

    if onscreen:
       if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
          print("Show confusion matrix on display")

          plt.show()
       else:
          print("Non-interactive backend; figure saved but not shown.")
          plt.close(fig)
    else:
       plt.close(fig)

def display_activations(input, label, activations, layer_names, figure_path, figure_name, figure_format, onscreen=True):
    fig = plt.figure(layout='constrained', figsize=(10, 8))
    subfigs = fig.subfigures(1, len(activations)+1) # layers + input , figsize=(row_size*2.5,len(model.layers)*1.5))

    subfigs[0].subplots(1, 1)
    subfigs[0].suptitle('Label: {}'.format(label))
    axs = subfigs[0].get_axes()
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].imshow(input, cmap='gray_r')

    for layer_index in range(0,len(activations)):
        print("layer:" +str(layer_index))
        print(activations[layer_index].shape[-1])
        subfigs[layer_index+1].suptitle(layer_names[layer_index])
        subfigs[layer_index+1].subplots(activations[layer_index].shape[-1],1)

    for layer_index in range(0,len(activations)):
        print(activations[layer_index].shape)
        #range(0,activations.shape[-1]):
        axs = subfigs[layer_index+1].get_axes()
        for plane_index in range(0,activations[layer_index].shape[-1]):
            plane = activations[layer_index][0,:, :, plane_index]
            axs[plane_index].set_xticks([])
            axs[plane_index].set_yticks([])
            axs[plane_index].imshow(plane, cmap='gray_r')

    fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
    if onscreen:
       if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
          plt.show()
       else:
          print("Non-interactive backend; figure saved but not shown.")
          plt.close(fig)
    else:
       plt.close(fig)

def display_weights_column(weights, layer_names,figure_path,figure_name,figure_format,onscreen=True):
    n_layers_with_weights = 0
    for layer_index in range(0, len(weights)):
        layer_weights = weights[layer_index]
        if len(layer_weights) > 0:
            n_layers_with_weights += 1

    fig = plt.figure(figsize=(30, 15), frameon=False)

    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
    subfigs = fig.subfigures(1, n_layers_with_weights)

    if not isinstance(subfigs, np.ndarray):
        subfigs = np.array([subfigs])

    layer_index_with_weights = 0
    print("Number of layers: "+str(len(weights)))
    for layer_index in range(0, len(weights)):
        layer_weights = weights[layer_index]

        print("layer:" +str(layer_index))
        # only weights (0) no biases (1)
        if len(layer_weights) > 0 and len(layer_weights[0].shape) > 1:
            print(" weights shape ", layer_weights[0].shape)

            #subfigs[layer_index_with_weights].suptitle(layer_names[layer_index])
            #subfigs[layer_index_with_weights].tight_layout()
            # squeeze=False squeezing at all is done: the returned Axes object is always a 2D array containing
            axs = subfigs[layer_index_with_weights].subplots(layer_weights[0].shape[-2], layer_weights[0].shape[-1], squeeze=False)#, sharex=True, sharey=True)
            subfigs[layer_index_with_weights].subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
            print(axs.shape)
            for i in range(0, layer_weights[0].shape[-2]):
                for j in range(0, layer_weights[0].shape[-1]):
                    w = layer_weights[0]
                    axs[i,j].imshow(w[:,:,i,j], cmap='gray_r', interpolation='nearest')
                    axs[i,j].axis("off")

            layer_index_with_weights += 1
    fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)

    if onscreen:
       if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
          plt.show()
       else:
          print("Non-interactive backend; figure saved but not shown.")
          plt.close(fig)
    else:
       plt.close(fig)


def display_loss_function(history,figure_path,figure_name,figure_format,onscreen=True):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    fig = plt.figure()
    plt.plot(epochs, loss, color='red', label='Training loss')
    plt.plot(epochs, val_loss, color='green', label='Validation loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
    if onscreen:
       print("Show loss on display")
       if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
          plt.show()
       else:
          print("Non-interactive backend; figure saved but not shown.")
          plt.close(fig)
    else:
       plt.close(fig)

# loading the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# normalizing the data 
X_train /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = to_categorical(y_train, n_classes)
Y_test = to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)
n_cnn1planes = 5
n_cnn1kernel = 3
n_poolsize = 1
n_cnn_layers = 1

# Stride defines the step size at which the filter moves across the input during convolution. 
# A larger stride results in a reduction of the spatial dimensions of the output feature map. 
# Stride can be adjusted to control the level of downsampling in the network.
# Stride is a critical parameter for controlling the spatial resolution of the feature maps and influencing the receptive field of the network.
n_strides = 1
n_dense = 100
dropout = 0.3

n_epochs=5

model_name = 'CNN_Handwritten_OCR_CNN'+str(n_cnn1planes)+'_KERNEL'+str(n_cnn1kernel)+'_Epochs' + str(n_epochs)
#figure_format='svg'
figure_format='png'
figure_path='./'
log_path='./log'


if not os.path.exists('task1_results'):
    os.makedirs('task1_results')

def build_model(n_layers, n_planes, kernel_size=3, pool_size=2):
    model = Sequential()

    # Layer 1 (Always included)
    model.add(Conv2D(n_planes, kernel_size=(kernel_size, kernel_size),
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(pool_size, pool_size)))

    # Layer 2 (Optional)
    if n_layers >= 2:
        # Usually we increase filters deeper in the network (e.g., *2)
        model.add(Conv2D(n_planes * 2, kernel_size=(kernel_size, kernel_size), activation='relu'))
        model.add(MaxPool2D(pool_size=(pool_size, pool_size)))

    # Layer 3 (Optional)
    if n_layers >= 3:
        model.add(Conv2D(n_planes * 4, kernel_size=(kernel_size, kernel_size), activation='relu'))
        model.add(MaxPool2D(pool_size=(pool_size, pool_size)))

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

layer_options = [1, 2, 3]
plane_options = [5, 15, 25, 35]

results_summary = []

for n_layers in layer_options:
    for n_planes in plane_options:

        # Define a unique name for this run
        model_name = f'Task1_L{n_layers}_P{n_planes}'
        print(f"\n--- Running Experiment: {model_name} ---")

        # Build
        try:
            model = build_model(n_layers, n_planes)
        except ValueError as e:
            # Skip invalid topologies (e.g. 3 layers might reduce image size to <0 if not careful)
            print(f"Skipping {model_name} due to size issues: {e}")
            continue

        # Compile
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

        # Train
        history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=128, epochs=5, verbose=1)

        # Save Loss Plot
        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f'Loss: {model_name}')
        plt.legend()
        plt.savefig(f'task1_results/{model_name}_loss.png')
        plt.close()

        # Record final validation accuracy
        final_val_acc = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        train_loss = history.history['loss'][-1]
        results_summary.append(f"{model_name}: Val Acc = {final_val_acc:.4f}, Val Loss = {val_loss}, Train Loss = {train_loss}")

# Print summary at the end
print("\n--- Summary of Task 1 ---")
for line in results_summary:
    print(line)




