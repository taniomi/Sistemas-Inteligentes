# Annotations

### Modeling
1. How many layers? How is the model affected by too many layers?
    The right answer is “try different variants and see what works”. I usually work with 2 Dense hidden layers with Dropout, a technique
that reduces overfitting by randomly setting inputs to 0. Hidden layers are useful to overcome the non-linearity of data, so if you don’t need non- linearity then you can avoid hidden layers. Too many hidden layers will lead to overfitting.

2. How many neurons?
    The number of hidden neurons should be between the size of the input layer and the size of the output layer. My rule of thumb is (number of inputs + 1 output)/2.

3. What activation function?
    There are many and we can’t say that one is absolutely better. Anyway, the most used one is ReLU, a piecewise linear function that returns the output only if it’s positive, and it is mainly used for hidden layers. Besides, the output layer must have an activation compatible with the expected output. For example, the linear function is suited for regression problems while the Sigmoid is frequently used for classification.

4. How much data should be for training and how much for validation?

5. What should be the batch size?

6. Which loss function? (cost)

7. How to optimize? (optimize = adjust)
    Through backpropagation, but which function defines the backpropagation? How is the gradient descent done?

8. What metrics? (what are metrics?)

### Activation Function
[Keras Activations - Tensor Flow DOCS](https://www.tensorflow.org/api_docs/python/tf/keras/activations)
[Activation Function - Wikipedia](https://en.wikipedia.org/wiki/Activation_function)

### Loss
[Keras Losses - Tensor Flow DOCS](https://www.tensorflow.org/api_docs/python/tf/keras/losses)

### Batch sizes
The backpropagation and the consequent parameters update happen every batch. An
epoch is one pass over the full training set. So, if you have 100 observations and
the batch size is 20, it will take 5 batches to complete 1 epoch. The batch size
should be a multiple of 2 (common: 32, 64, 128, 256) because computers usually
organize the memory in power of 2. I tend to start with 100 epochs with a batch
size of 32.

Moreover, it’s good practice to keep a portion of the data (20%-30%) for validation. 
In other words, the model will set apart this fraction of data to evaluate the loss 
and metrics at the end of each epoch, outside the training.

### Implementing
- Inicialmente a regressão foi realizada com 100 épocas. Mas a velocidade de aprendizado não foi a esperada, então o número de épocas foi ajustado para 1000 épocas, e os gráficos de aprendizado mostraram diminuição significativa na perda.
