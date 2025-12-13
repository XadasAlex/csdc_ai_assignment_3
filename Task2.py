import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import SGD  # Wichtig: SGD importieren
from tensorflow.keras.utils import to_categorical

if not os.path.exists('task2_results'):
    os.makedirs('task2_results')

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)

# building the model that performed best in task 1: L2_P25
def build_winning_model():
    planes = 25

    model = Sequential()
    # layer 1
    model.add(Conv2D(planes, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    # layer 2
    model.add(Conv2D(planes * 2, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dropout(0.3))  # Dropout lassen wir vorerst konstant
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


# Exercise: Vary between [0.001, 0.01], evaluate min 4 values.
learning_rates = [0.001, 0.004, 0.007, 0.01]

results_summary = []

for lr in learning_rates:
    model_name = f'Task2_LR_{lr}'
    print(f"\n--- Running Experiment: {model_name} ---")

    model = build_winning_model()

    # as specified use the sgd optimizer with the respective learn rates

    optimizer = SGD(learning_rate=lr)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    # Training
    history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=128, epochs=5, verbose=1)

    # Save Loss Plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss: {model_name} (SGD)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'task2_results/{model_name}_loss.png')
    plt.close()

    # Save Final Metrics
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    results_summary.append(f"{model_name}: Val Loss = {final_val_loss:.4f}, Val Acc = {final_val_acc:.4f}, Training Loss = {final_train_loss:.4f}")

print("\n--- Summary of Task 2 ---")
for line in results_summary:
    print(line)