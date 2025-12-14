Group members: Alexander Gebhart, Sebastian Pfeiffer, Filip Lukic

### Task 1: Topology

1. reﬂect on the function of the convolution and pooling layers for 2D classiﬁcation

**Convolutional Layers:** The primary function of these layers is feature extraction. By applying filters (kernels) to the input image, the layer activates upon detecting specific local patterns, such as edges, lines, or curves, regardless of their position in the image. 

**Pooling Layers:** These layers perform down-sampling (dimensionality reduction). They play a critical role by reducing the computational complexity and making the model robust against small variations in the position of features (translation invariance).

2. modify the feature extraction by using 1, 2 and 3 pairs of convolution and pooling layer

Introduced a `build_model` function so that the process can be iterated easily.

```python

def build_model(n_layers, n_planes, kernel_size=3, pool_size=2):  
    model = Sequential()  
  
    model.add(Conv2D(n_planes, kernel_size=(kernel_size, kernel_size),  
                     activation='relu', input_shape=(28, 28, 1)))  
    model.add(MaxPool2D(pool_size=(pool_size, pool_size)))  
  
    if n_layers >= 2:  

        model.add(Conv2D(n_planes * 2, kernel_size=(kernel_size, kernel_size), activation='relu'))  
        model.add(MaxPool2D(pool_size=(pool_size, pool_size)))  
  

    if n_layers >= 3:  
        model.add(Conv2D(n_planes * 4, kernel_size=(kernel_size, kernel_size), activation='relu'))  
        model.add(MaxPool2D(pool_size=(pool_size, pool_size)))  
  
    model.add(Flatten())  
    model.add(Dropout(0.3))  
    model.add(Dense(100, activation='relu'))  
    model.add(Dense(10, activation='softmax'))  
  
    return model

```

Like so: 

```python

for n_layers in layer_options:  
    for n_planes in plane_options:
	    # ..
		#  ...
		# ...
		model = build_model(n_layers, n_planes)
		

```

3. vary the number of planes/feature maps between [5, 40] (evaluate min. 4 values)

To determine the optimal network topology, we conducted a systematic evaluation modifying two key parameters:

**Depth:** We tested 1, 2, and 3 pairs of Convolutional and Pooling layers.

**Width:** We varied the number of feature planes (filters) in the first layer between [5, 15, 25, 35].

This resulted in 12 distinct configurations (3 depths×4 widths). We analyzed the Loss Function and Accuracy for both training and validation sets for each configuration.

Upon reviewing the aggregated results, the configuration with **2 Layers and 35 Feature Planes (L2_P35)** initially appeared to be the strongest candidate. It achieved a very high validation accuracy of **98.78%** and the lowest training loss of **0.0610**, suggesting excellent learning capability.


![[Task1_L2_P35_loss.png]]


4. change the name of the model in the model_name variable to identify the output ﬁles for model loss function, accuracy, and classiﬁcation report

```python

if not os.path.exists('task1_results'):  
    os.makedirs('task1_results')

```

```python

layer_options = [1, 2, 3]  
plane_options = [5, 15, 25, 35]  
  
results_summary = []  
  
for n_layers in layer_options:  
    for n_planes in plane_options:  
  
        model_name = f'Task1_L{n_layers}_P{n_planes}'  
        print(f"Running Experiment: {model_name}")

```

```python
	plt.figure()  
	plt.plot(history.history['loss'], label='Train Loss')  
	plt.plot(history.history['val_loss'], label='Val Loss')  
	plt.title(f'Loss: {model_name}')  
	plt.legend()  
	plt.savefig(f'task1_results/{model_name}_loss.png')  
	plt.close()
```

![[Pasted image 20251213105835.png]]

```python

	final_val_acc = history.history['val_accuracy'][-1]  
	val_loss = history.history['val_loss'][-1]  
	train_loss = history.history['loss'][-1]  
	results_summary.append(f"{model_name}: Val Acc = {final_val_acc:.4f}, Val Loss = {val_loss}, Train Loss = {train_loss}")

```

![[Pasted image 20251213110123.png]]

5. analyse the loss function for both training and validation data for the diﬀerent conﬁgurations &  note observations in the report together with the ﬁgures of the loss functions

![[Pasted image 20251213110233.png]]

The training loss and validation loss spiked when we introduced too much complexity: take a look at `L3_P5`. To put this into perspective we've sorted all the values by the categories `validatio_accuracy`, `validation_loss`, `training_loss`

### Validation Accuracy Sorted

| Task1_L3_P5:  | 0.9502 |
| ------------- | ------ |
| Task1_L1_P5:  | 0.9723 |
| Task1_L1_P15: | 0.9782 |
| Task1_L1_P35: | 0.9787 |
| Task1_L3_P15: | 0.9792 |
| Task1_L1_P25: | 0.9802 |
| Task1_L3_P25: | 0.9805 |
| Task1_L3_P35: | 0.9825 |
| Task1_L2_P5:  | 0.9832 |
| Task1_L2_P15: | 0.9852 |
| Task1_L2_P35: | 0.9878 |
| Task1_L2_P25: | 0.9893 |

### Validation Loss Sorted

| Task1_L2_P25: | 0.0455061532557011 |
| ------------- | ------------------ |
| Task1_L2_P35: | 0.0469560325145721 |
| Task1_L2_P15: | 0.0491616502404213 |
| Task1_L3_P35: | 0.0576742403209209 |
| Task1_L2_P5:  | 0.0592364259064198 |
| Task1_L3_P25: | 0.0623621456325054 |
| Task1_L3_P15: | 0.0721407756209374 |
| Task1_L1_P35: | 0.0747978985309601 |
| Task1_L1_P25: | 0.0765363946557045 |
| Task1_L1_P15: | 0.0784859210252762 |
| Task1_L1_P5:  | 0.0964559614658356 |
| Task1_L3_P5:  | 0.17912170290947   |
### Train Loss Sorted

| Task1_L2_P35: | 0.0610844232141972 |
| ------------- | ------------------ |
| Task1_L2_P25: | 0.0676925703883171 |
| Task1_L2_P15: | 0.0793734565377235 |
| Task1_L1_P35: | 0.0877408161759377 |
| Task1_L3_P35: | 0.0998020619153976 |
| Task1_L1_P25: | 0.099833071231842  |
| Task1_L3_P25: | 0.10835374891758   |
| Task1_L1_P15: | 0.114117838442326  |
| Task1_L2_P5:  | 0.128324434161186  |
| Task1_L3_P15: | 0.138650983572006  |
| Task1_L1_P5:  | 0.154231712222099  |
| Task1_L3_P5:  | 0.393651574850082  |

#### IMPORTANT OVERFITTING

|               | Val_Acc | Val_Loss | Train_Loss |
| ------------- | ------- | -------- | ---------- |
| Task1_L2_P15: | 0.9852  | 0.0491   | 0.0793     |
| Task1_L2_P25: | 0.9893  | 0.0455   | 0.0676     |
| Task1_L2_P35: | 0.9878  | 0.0469   | 0.0610     |

Analysis of the Loss Functions: the highest accuracy did not correspond to the best generalization. The assumption that L2_P35 is the most competent and promising configuration was wrong because we didn't factor in the Val_Loss metric  
#### To put it all together:

Comparing the layer depths, it became evident that a 3rd layer introduced unnecessary complexity. As seen in the _Task1_L3_ models, the performance dropped significantly compared to L2. The third pooling step likely reduced the spatial resolution too drastically, causing the loss of essential information required for classification.

**Analysis of Width (Overfitting):** Comparing the width for the 2-Layer models revealed a critical insight regarding overfitting:

**L2_P35 (35 Planes):** While the training loss continued to drop, the validation loss began to rise slightly at the end of the training (0.0469). This divergence indicates that the model was beginning to memorize the training data (overfitting) rather than generalizing.

**L2_P25 (25 Planes):** This configuration achieved the **lowest validation loss (0.0455)** of all models tested. Crucially, the validation loss curve remained stable and did not increase, indicating superior generalization compared to the P35 model.

6. Conclusion

Based on the analysis of the validation loss, we revised our initial assessment. We concluded that **2 Layers with 25 Feature Planes** is the optimal topology. It balances model complexity with generalization power, avoiding the overfitting observed with 35 planes while outperforming the simpler 1-layer and 15-plane models.


### Task 2: Learning Rate

1. reﬂect on the function of the learning rate as parameter for optimization algorithms

The **Learning Rate** (η) is arguably the most critical hyperparameter in training neural networks. It determines the **step size** at which the optimizer (SGD) updates the model's weights during gradient descent.

- **Too Low:** The model learns effectively but extremely slowly. It may get stuck in local minima or require an impractical number of epochs to converge (Underfitting).
    
- **Too High:** The model may "overshoot" the optimal minimum, causing the loss function to oscillate or even diverge (instability).
    
- **Goal:** To find a value that allows for rapid convergence while maintaining stability.

2. use the Stochastic-Gradient-Descent (SGD) algorithm as optimizer

```python

model = build_winning_model()  
  
# as specified use the sgd optimizer with the respective learn rates  
  
optimizer = SGD(learning_rate=lr)  
  
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)  
  
# Training  
history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=128, epochs=5, verbose=1)

```

3. vary the learning rate between [0.001,0.01] (evaluate min. 4 values)

For this experiment, we fixed the topology to our winning model from Task 1 (**L2_P25**). We utilized the **Stochastic Gradient Descent (SGD)** optimizer and systematically varied the constant learning rate across four distinct values: `[0.001, 0.004, 0.007, 0.01]`.

```python
earning_rates = [0.001, 0.004, 0.007, 0.01]  
  
results_summary = []  
  
for lr in learning_rates:
	# ...
```

4. change the name of the model in the model_name variable to identify the output ﬁles for model loss function, accuracy, and classiﬁcation report

```python
	model_name = f'Task2_LR_{lr}'
	# ...
	
	plt.figure()  
	plt.plot(history.history['loss'], label='Train Loss')  
	plt.plot(history.history['val_loss'], label='Val Loss')  
	plt.title(f'Loss: {model_name} (SGD)')  
	plt.xlabel('Epochs')  
	plt.ylabel('Loss')  
	plt.legend()  
	plt.savefig(f'task2_results/{model_name}_loss.png')  
	plt.close()
```

5. analyse the loss function for both training and validation data for each learning rate & note observations in the report together with the ﬁgures of the loss functions

![[Pasted image 20251213121321.png]]

Plotted that in `LibreCalc`

![[Pasted image 20251213121558.png]]

![[Pasted image 20251213121548.png]]


| **Experiment**     | **Learning Rate (η)**  | **Val Acc** | **Val Loss** | **Train Loss** | **Interpretation**                                                                                              |
| ------------------ | ---------------------- | ----------- | ------------ | -------------- | --------------------------------------------------------------------------------------------------------------- |
| **Task2_LR_0.001** | 0.001                  | 0.7665      | 1.1487       | 1.5089         | Too slow (underfitting): The loss is too high and the accuracy is low. The model didn't learn much in 4 Epochs. |
| **Task2_LR_0.004** | 0.004                  | 0.9375      | 0.2284       | 0.4054         | Better: The loss decreased and the model started to learn effectively                                           |
| **Task2_LR_0.007** | 0.007                  | 0.9592      | 0.155        | 0.2751         | Very Good: Better balance between learn rate and precision                                                      |
| **Task2_LR_0.01**  | 0.01                   | **0.9653**  | **0.1289**   | **0.2181**     | Best: Val_Loss is the lowest while the accuracy is the highest.                                                 |
**Observation 1: The Trap of Low Learning Rates (Underfitting)**

- **Configuration:** `Task2_LR_0.001`
    
- Analysis: As seen in the graphs and the table, this configuration resulted in a massive Validation Loss of 1.1487 and a low Accuracy of 76.65%. Crucially, the Training Loss (1.5089) remained extremely high.
    
- Conclusion: This is a clear case of underfitting. The step size was so small that the model could barely descend the loss gradient within the allotted 5 epochs. While it _is_ learning, it is doing so inefficiently.
    

**Observation 2: Progressive Improvement**

- Configurations: `Task2_LR_0.004` and `Task2_LR_0.007`
    
- Analysis: Increasing the rate showed immediate benefits. At **LR 0.004**, the validation loss dropped significantly to 0.2284. At LR 0.007, it improved further to 0.1550. The steepness of the loss curves in the diagrams confirms that the model is converging much faster.
    

**Observation 3: The Optimum**

- Configuration: `Task2_LR_0.01`
    
- Analysis: This setting achieved the best performance across all metrics.
    
    - Lowest Validation Loss: 0.1289
        
    - Highest Accuracy: 96.53%
        
    - Lowest Training Loss: 0.2181
        
- **Conclusion:** The loss curve for 0.01 shows a steep, healthy descent without signs of instability or oscillation. It provides the best balance between learning speed and precision for this specific topology.

#### Conclusion for Task 2

Based on the comparative analysis, a constant learning rate of **0.01** is the optimal choice for this dataset and topology. It eliminates the underfitting observed at lower rates and achieves high accuracy quickly. This value will serve as the **baseline (initial learning rate)** for the Learning Rate Schedules in Task 3.


# NOTE THAT I EXPORTED A PDF FILE THAT CONTAINS OTHER SCREEN SHOTS FOR DOCUMENTATION PURPOSES - I DIDNT WANT TO ADD SCREENSHOTS TO THE REPO FOR THAT PURPOSE.
# aissngment3.ods contains the spreadsheet

### Task 3: Learning Rate Schedules

1. reflect on the function of the learning rate as parameter for optimization algorithms and how alternative learning rate schedules effect the optimization

When an optimization algorithm adjusts the weights, it needs a step size to determine the weight change. This step size is determined by the learning rate and significantly impacts the speed and stability of the optimization.

If the learning rate is too low, progress is slow, if it is too high, it leads to instability (oscillation and divergence are possible).

When using learning rate schedules, the learning rate changes during training. It typically starts with high values ​​to accelerate progress. The values ​​then gradually decrease, making the optimization increasingly stable as it approaches the ideal weight.

Learning rate schedules thus serve to accelerate and stabilize the training.

2. use the Stochastic-Gradient-Descent (SGD) algorithm as optimizer

See task 2.2.

3. replace the original constant learning rate by a learning rate schedule

This has library has to be imported:

```python
from tensorflow.keras.optimizers.schedules import ExponentialDecay
```

Now, the learning rate schedule can be defined and used:

```python
learning_rate = ExponentialDecay(initial_learning_rate=1e-2,
								 decay_steps=n_epochs,
								 decay_rate=0.9)

optimizer = SGD(learning_rate=learning_rate)
```

4. change the name of the model in the model_name variable to identify the output files for
model loss function, accuracy, and classification report

See task 2.4.

5. analyse the loss function for both training and validation data for each learning rate

With the initial learning rate of 1e-2 specified in the task description, the following result is achieved:

[Task3_initLR_1e2_loss.png]

| Training Loss | Val Loss | Val Acc |
| -------- | -------- | -------- |
| 2.2573 | 2.2430 | 0.2567 |

As can be seen, the training loss only decreases after the first epoch, the loss value then remains consistently high. The model is therefore barely learning.

The validation loss also hardly changes, but is significantly lower than the training loss.

The reason for this stagnation is that an initial learning rate of 1e-2 is too high to achieve any significant optimization within five epochs.

6. extend the experimental scope by varying the initial learning rate between [0.001,0.1] and
the type of schedule (evaluate min. 4 values)

The following learning rates are used:
- 0.001
- 0.003
- 0.05
- 0.1

The following schedule types are used:
- Exponential Decay
- Inverse Time Decay
	- requires
	  ```python
	  from tensorflow.keras.optimizers.schedules import InverseTimeDecay
	  ```

For the "exponential decay" type, this code is used:

```python
learning_rate = ExponentialDecay(initial_learning_rate=0.001,
 								 decay_steps=n_epochs,
								 decay_rate=0.9)
```

For the "inverse time decay" type, this code is used:

```python
learning_rate = InverseTimeDecay(initial_learning_rate=0.001,
 								 decay_steps=n_epochs,
								 decay_rate=0.9)
```

7. note observations in the report together with the figures of the loss functions

| Initial LR | Type | Training Loss | Val Loss | Val Acc |
| ------- | ---------------- | -------- | -------- | -------- |
| 0.001 | Exponential Decay | 2.2972 | 2.2894 | 0.0712 |
| 0.001 | Inverse Time Decay | 2.2848 | 2.2811 | 0.1005 |
| 0.003 | Exponential Decay | 2.2895 | 2.2862 | 0.1600 |
| 0.003 | Inverse Time Decay | 2.2971 | 2.2974 | 0.1258 |
| 0.05 | Exponential Decay | 1.7250 | 1.6385 | 0.6892 |
| 0.05 | Inverse Time Decay | 1.9169 | 1.8354 | 0.6517 |
| 0.1 | Exponential Decay | 0.6364 | 0.4136 | 0.8955 |
| 0.1 | Inverse Time Decay | 1.0121 | 0.7662 |  0.8405 |

[ Alle Bilder vom Task3 Ordner ]

The results show that low learning rates (0.01 and 0.003) result in nearly constant loss functions. The model therefore learns very little, regardless of the schedule type.

At higher learning rates (0.05 and 0.1), the exponential decay schedule type achieves the best results. The inverse time decay schedule type, on the other hand, results in stable but slower learning behavior.

This demonstrates that both the chosen learning rate and the schedule type used have a significant impact on the loss function.

### Task 4: Optimizers

1. reflect on the function of the momentum term if applied to the Stochastic-Gradient-Descent
(SGD) algorithm

When using the momentum term, previous weight shifts are also considered to determine the direction of movement.

The goal is to stabilize the training by reducing strong oscillation of weights.

One can visualize it like this:

Someone is on a hill (a high point on the loss function) and they want to descend into the valley (the desired ideal value). Before each step, they ask themselves: "Does it feel like I should go left or right?" Without the momentum term, they ignore all previous steps and move in the direction that seems most appropriate based on their current position. With the term, however, they also consider previous steps and say, for example: "If I've already taken five steps to the left and now right feels better, I'll maintain my current direction for now and then see what happens." This way, they don't constantly switch back and forth between left and right, and their direction of movement remains more consistent.

The momentum term therefore improves both the stability and the speed of the training.

2. enhance the Stochastic-Gradient-Descent (SGD) algorithm with the momentum term

```python
learning_rate=0.001
momentum=0.9
optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
```

3. change the name of the model in the model_name variable to identify the output files for
model loss function, accuracy, and classification report

See task 2.4.

4. vary the learning rate between [0.001,0.01] (evaluate min. 4 values)

The following learning rates are used:
- 0.001
- 0.003
- 0.007
- 0.01

5. analyse the loss function for both training and validation data for each learning rate

The following table shows the results:

| LR | Training Loss | Val Loss | Val Acc |
| ------- | -------- | -------- | -------- |
| 0.001 | 0.2094 | 0.1228 | 0.9663 |
| 0.003 | 0.1151 | 0.0686 | 0.9805 |
| 0.007 | 0.0771 | 0.0571 | 0.9832
| 0.01 | 0.0655 | 0.0493 | 0.9870 |

[ alle Bilder vom Task4 Ordner ]

6. note observations in the report together with the figures of the loss functions

It can be seen that using the momentum term leads to faster and more stable convergence. Loss decreases significantly within just a few epochs.

Furthermore, it can be observed that increasing the learning rate results in continuously improving performance. No instabilities occur, thus, momentum has a stabilizing effect.

The validation loss is always lower than the training loss, indicating no overfitting.

Therefore, it is clear that the momentum term improves optimization and allows for the efficient and stable use of higher learning rates.

### Task 5: Dropout Layer (Regularization)

1. effects of introducing dropout layers (particularly between convolutional layers)

To analyze the effect of dropout, we trained the same CNN topology both with and without dropout layers.

Dropout was inserted between convolutional blocks (after max pooling layers) with a value of '0.3'.

The experiments were made with 5 epochs and 20 epochs.

5 epochs:

![[Task5_WithoutDropoutsLoss_1_loss.png]]

![[Task5_DropoutsLoss_1_loss.png]]

20 epochs:

![[Task5_WithoutDropoutsLoss_2_loss.png]]

![[Task5_DropoutsLoss_2_loss.png]]

In the short training run (5 epochs), training and validation loss decreased simultaneously and remained close to each other, indicating no immediate overfitting. In this scenario, the effect of dropout is not yet clearly visible.

When extending the training to 20 epochs, the model without dropout shows a continuously decreasing training loss, while the validation loss stagnates and exhibits small changes. This behavior indicates overfitting, as the model increasingly memorizes the training data without improving its generalization capability.

With dropout enabled, the training loss remains higher due to the random deactivation of feature maps during training. However, the validation loss decreases more smoothly and remains stable over the entire training process. Thats a good indication for generalization and for reduced overfitting.

2. insert a dropout layer at appropriate position between instructions that compose the current topology

Dropout layers were inserted after the max pooling layers between convolutional blocks. This position was chosen to regularize feature maps while preserving essential spatial information extracted by the convolutional layers.

3. vary the dropout rate between [0.2,0.5] for all introduced dropout layers (evaluate min. 4 values)

Prepare list of dropouts and the build_model function:

```python
def build_model(dropout_rate=0.3):
    ...
    # layer 1
    model.add(Conv2D(planes, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))    #dropout
    # layer 2
    model.add(Conv2D(planes * 2, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))    #dropout
	...

dropout_rates = [0.2, 0.3, 0.4, 0.5]
```

4. change the name of the model in the model_name variable to identify the output files for model loss function, accuracy, and classification report

```python
dropout_rates = [0.2, 0.3, 0.4, 0.5]

for dr in dropout_rates:
	...
    model = build_model(dropout_rate=dr)
    model_name = f'Task5_DR{dr}'
	...
```

5. analyse the loss function for both training and validation data for each dropout layer insertion and dropout rate

Dropout rate = 0.2

![[Task5_DR0.2_loss.png]]

Dropout rate = 0.3

![[Task5_DR0.3_loss.png]]

Dropout rate = 0.4

![[Task5_DR0.4_loss.png]]

Dropout rate = 0.5

![[Task5_DR0.5_loss.png]]

For each evaluated dropout rate, the training and validation loss curves were plotted and analyzed. The loss curves show a monotonic decrease of both training and validation loss across all epochs for each configuration. The training loss decreases faster in the early epochs, while the validation loss follows a similar trend and stabilizes towards the end of training.

6. note observations in the report together with the figures of the loss functions

Lower dropout rates lead to faster convergence and lower training loss, but show slightly larger gaps between training and validation loss. Higher dropout rates increase training loss and slow down convergence, indicating stronger regularization.

### Task 6: Final accuracy evaluation

1. evaluate the accuracy report and the confusion matrix of the test data classified using the best model

Confusion Matrix:

![[best_model_confusion_matrix.png]]

Accuracy report:

              precision    recall  f1-score   support

           0     0.9909    0.9949    0.9929       980
           1     0.9982    0.9938    0.9960      1135
           2     0.9923    0.9961    0.9942      1032
           3     0.9941    0.9960    0.9951      1010
           4     0.9969    0.9878    0.9923       982
           5     0.9899    0.9877    0.9888       892
           6     0.9885    0.9896    0.9890       958
           7     0.9941    0.9883    0.9912      1028
           8     0.9877    0.9887    0.9882       974
           9     0.9833    0.9931    0.9882      1009

    accuracy                         0.9917     10000
   	macro avg    0.9916    0.9916    0.9916     10000
	weighted avg 0.9917    0.9917    0.9917     10000

