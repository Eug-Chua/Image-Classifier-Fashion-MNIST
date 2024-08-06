# Overview of the Project
In this notebook, we implement an image classifier using a deep learning framework to classify images from the Fashion MNIST dataset. The objective is to accurately classify 28x28 grayscale images into one of 10 fashion categories. The dataset comprises 60,000 training images and a test set of 10,000 images.

# Dataset Description
The Fashion MNIST dataset is a collection of grayscale images representing 10 distinct fashion categories.

Each image is 28x28 pixels. The categories are as follows:

| Label | Description |
|-------|-------------|
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

# Model Architecture
We constructed a Convolutional Neural Network (CNN) using Keras' Sequential API. The model comprises the following 12 layers:
1. `Conv2D`: A convolutional layer with 64 filters of size 3x3.
    - Slides filters over the input image, computing dot products at each position to create feature maps.
2. `BatchNormalization`: Normalizes and scales the output of the previous layer.
3. `MaxPooling2D`: Reduces the spatial dimensions of the feature maps by taking the maximum value over a 2x2 window.
4. `Conv2D`: A convolutional layer with 128 filters of size 3x3.
    - Extracts more complex features by combining lower-level features detected by earlier layers.
5. `BatchNormalization`: Normalizes and scales the output of the previous layer.
6. `MaxPooling2D`: Further reduces the spatial dimensions of the feature maps.
7. `Dropout`: Randomly drops a proportion of the input units to prevent overfitting.
8. `Flatten`: Flattens the 2D feature maps into a 1D vector for transitioning to fully connected layers.
9. `Dense`: A fully connected layer with ReLU activation.
    - Learns complex representations by connecting all neurons of the previous layer.
10. `BatchNormalization`: Normalizes and scales the output of the previous layer.
11. `Dropout`: Randomly drops a proportion of the input units to prevent overfitting.
12. `Dense`: The final fully connected layer with softmax activation.
    - Converts the output into a probability distribution over the 10 classes.

# Hyperparameter Tuning
To enhance model performance, we tuned several hyperparameters using Keras Tuner's `RandomSearch`:
- Layer 7 `Dropout` - proportion of data to drop off: `0.2`, `0.3`, `0.4`, `0.5`
- Layer 9 `Dense` - number of neurons: `256`, `320`, `384`, `448`, `512`
- Layer 11 `Dropout` - proportion of data to drop off: `0.2`, `0.3`, `0.4`, `0.5`
- `learning_rate` - ranges from 0.0001 to 0.01; value will beinputted into the `Adam` optimizer

# Cross-Validation and Training
To prevent overfitting and ensure robust performance, we employed 4-fold cross-validation during hyperparameter tuning. We set the number of epochs to 10 and `max_trials` to 3, resulting in 12 model trainings (3 trials * 4 folds) and a total of 120 epoch trainings across the entire tuning process.

# Model Evaluation
After training the model, we achieved an accuracy score of 90.7% on the training data. The best hyperparameters identified were:
- Layer 7 `Dropout` - proportion of data to drop off: `0.4`
- Layer 9 `Dense` - number of neurons: `320`
- Layer 11 `Dropout` - proportion of data to drop off: `0.4`
- `learning_rate` - 0.000263474572751995

The final test score was 91.6%, comparable to the validation accuracy, indicating low evidence of overfitting. Precision and recall scores were both 92%.

# Detailed Performance
- Lower Performance: The model struggled more with items related to the upper body, such as `T-shirt/top`, `Pullover`, `Coat`, and `Shirt`. The accuracy for `Shirt` was notably lower, with an F1 score of 75%.
- Higher Performance: The model performed exceptionally well on items such as `Trouser`, `Sandal`, `Bag`, and `Ankle Boot`, with F1 scores of ≥97%.

# Visualizations
The confusion matrix visualized the number of correct classifications and misclassifications for each item, indicating where the model performed well and where it struggled. The ROC-AUC scores for each class were consistently high, with the lowest being 0.976 for Shirt and several others near perfect (>0.999), demonstrating the model's excellent performance in distinguishing between positive and negative instances for each class.

# Conclusion
Our CNN model demonstrated robust performance in classifying the Fashion MNIST dataset, achieving high precision, recall, F1 scores, and ROC-AUC values across all classes.

The 12-layer neural network architecture and thorough hyperparameter tuning significantly contributed to its effectiveness. The evaluation metrics indicate that the model is well-suited for this classification task with minimal overfitting.

Future improvements could focus on enhancing the model's ability to distinguish between similar upper body clothing items.