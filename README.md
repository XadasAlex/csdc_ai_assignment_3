
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

