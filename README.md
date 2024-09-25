# neural-network-challenge-1
This project is focused on developing a neural network model that predicts the likelihood of a student repaying their loans based on a set of financial, academic, and personal attributes. The goal is to build and evaluate a binary classification model using deep learning techniques in Python, specifically leveraging TensorFlow and Keras.

The project walks through the steps of loading the dataset, preprocessing the data, building the neural network, training the model, and evaluating its performance on test data.

## File Structure

```
neural-network-challenge-1/
├── LICENSE
├── README.md
├── student_loans_with_deep_learning.ipynb
└── student_loans.keras
```
* LICENSE: Contains the licensing information for the project.
* README.md: This readme file that provides an overview of the project.
* student_loans_with_deep_learning.ipynb: A Jupyter notebook that includes all the code for loading the data, preprocessing, building the neural network model, training, and evaluation (intended to be run in Google CoLab)
* student_loans.keras: The trained neural network model saved in Keras format.

## Data
The dataset used in this project contains student information across multiple categories, including:
* Financial information: payment history, financial aid score, and credit ranking.
* Academic performance: GPA ranking, study major, and time to completion.
* Personal attributes: cohort ranking and location parameter.
* Target Variable: The target variable is credit_ranking, which is binary (0 or 1) and indicates whether a student is likely to repay their loans or not.


## Model Architecture
The neural network was implemented using the Keras Sequential API and consists of the following layers:
* Input Layer: 11 features representing the input data.
* Hidden Layer 1: A dense layer with 13 neurons and ReLU activation.
* Hidden Layer 2: A dense layer with 8 neurons and ReLU activation.
* Output Layer: A single neuron with sigmoid activation for binary classification.

Model Configuration:
* Loss Function: Binary Cross-Entropy
* Optimizer: Adam
* Metrics: Accuracy

## Training and Evaluation
The model was trained for 50 epochs with a batch size of 34 and a default 75-25 split between training and validation data. The training history shows that the model achieves an accuracy of close to 74% on the test data.

The confusion matrix and classification report were used to assess the model's performance in terms of precision, recall, and F1-score.

## Key Insights
The training process indicates some overfitting after approximately 10 epochs, as shown by the divergence between training and validation accuracy/loss. Early stopping or regularization techniques may help improve generalization.

## Installation and Usage
1. Clone this repository.
2. Install required dependencies:
    1. tensorflow
    2. pandas
    3. sklearn
    4. Path
    5. matplotlib
    6. seaborn
3. Open the Jupyter notebook 'student_loans_with_deep_learning.ipynb'
4. To load the saved model and make predictions:
```
from tensorflow.keras.models import load_model
model = load_model('student_loans.keras')
predictions = model.predict(X_test_scaled).round().astype('int32')
```

## Future Improvements
* Early Stopping: Implement early stopping to prevent overfitting during training.
* Hyperparameter Tuning: Experiment with different optimizers, learning rates, and network architectures to improve performance.

## License
This project is licensed under the Unlicense

## Author
David Kaplan