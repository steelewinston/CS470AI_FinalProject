import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def load_data():

    data = pd.read_excel('Documents\Spam_finder\spambase.xlsx')

    # Shuffle the row indices
    shuf_Indices = np.random.permutation(data.index)

    # Use the shuffled indices to reorder the rows of the DataFrame
    data_Shuf = data.iloc[shuf_Indices]

    return data_Shuf


def preprocess_data(data):

    # Assuming your data is already preprocessed and split into features (X) and labels (y)
    X = data.drop(columns=['spam', 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']).values
    y = data['spam'].values.reshape(-1, 1)
    
    # Split into training and testing sets (75% training, 25% testing)
    split_Ratio = 0.75
    split_Index = int(split_Ratio * len(data))
    x_Train, x_Test = X[:split_Index], X[split_Index:]
    y_Train, y_Test = y[:split_Index], y[split_Index:]
    
    return x_Train, x_Test, y_Train, y_Test


def init_Parameters(features):

    #array for holding feature weights 1 column of # of features 
    weights = np.zeros((features, 1))

    #intilize bias value
    bias = 0

    return weights, bias


def sigmoid(linear_Combination):

    # Clip linear_Combination to prevent overflow
    linear_Combination = np.clip(linear_Combination, -10, 10)

    # create the logistic equation 
    return 1 / (1 + np.exp(-linear_Combination))


def compute_loss(spam_Actual, spam_Pred):
    
    # Small epsilon value to avoid division by zero
    epsilon = 1e-15

    row_Count = spam_Actual.shape[0]
    loss = -1/row_Count * np.sum(spam_Actual * np.log(spam_Pred + epsilon) + (1 - spam_Actual) * np.log(1 - spam_Pred + epsilon))

    return loss


def gradient_descent(input_Data, labels, weights, bias, learning_Rate, num_Steps):

    num_Examples = input_Data.shape[0]  # Number of examples

    for index in range(num_Steps):
        # Forward propagation
        linear_Combination = np.dot(input_Data, weights) + bias
        predictions = sigmoid(linear_Combination)

        # Compute the loss
        cost = compute_loss(labels, predictions)

        # Backpropagation
        dw = 1/num_Examples * np.dot(input_Data.T, (predictions - labels))
        db = 1/num_Examples * np.sum(predictions - labels)

        # Update weights and bias
        weights -= learning_Rate * dw
        bias -= learning_Rate * db

        # Print the cost every 100 iterations
        if index % 100 == 0:
            print(f"Iteration {index}: Cost = {cost}")

    return weights, bias

def predict(X, weights, bias):

    linear_Combination = np.dot(X, weights) + bias
    predictions = sigmoid(linear_Combination)

    return predictions


# Step 6: Main Function
def main():
    # Load data
    data = load_data()
    
    # Preprocess data, 75/25 split for test and train
    # x holds the data while y hold the last spam column
    x_Train, x_Test, y_Train, y_Test = preprocess_data(data)

    # Initialize weights and bias, takes the number of features in x_Train
    weights, bias = init_Parameters(x_Train.shape[1])

    #calcualte and store costs at each iteration
    training_Weights, training_Bias = gradient_descent(x_Train, y_Train, weights, bias, learning_Rate = 0.01, num_Steps = 1000)

    # Predict labels for test data
    y_Pred = predict(x_Test, training_Weights, training_Bias)

    y_Pred_Binary = (y_Pred > 0.5).astype(int)

    # Evaluate performance
    print("Performance Metrics:")

    accuracy_Value = accuracy_score(y_Test, y_Pred_Binary)
    print(f"Accuracy: {accuracy_Value}")

    conf_Matrix = confusion_matrix(y_Test, y_Pred_Binary)
    tn, fp, fn, tp = conf_Matrix.ravel()
    print(f"False Positives: {fp}")
    print(f"True Positives: {tp}")

    auc_Value = roc_auc_score(y_Test, y_Pred)
    print(f"AUC: {auc_Value}")

if __name__ == "__main__":
    main()