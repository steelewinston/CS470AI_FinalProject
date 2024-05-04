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


# straight line distance betwwen two points
def euclidean_distance(data_PointOne, data_PointTwo):
    return np.sqrt(np.sum((data_PointOne - data_PointTwo)**2))


def knn_predict(x_Train, y_Train, x_Test, k=5):
    # Create list to hold distance 
    distances = []

    # Iterate over email lis length
    for index in range(len(x_Train)):

        # Calcualte the distance between the current point and test point
        dist = euclidean_distance(x_Test, x_Train[index])

        # Save the data tuple to the list
        distances.append((dist, y_Train[index]))
    
    # Sort the distances for the attribute in ascending order and 
    # Selects the first k neighbor elements
    distances = sorted(distances)[:k]

    # Convert data type to tuple for hashing
    labels = [tuple(dist[1]) for dist in distances]

    # Determin spam or not by finding frequent labels
    prediction = max(set(labels), key=labels.count)

    return prediction


def knn(x_Train, y_Train, x_Test, k=5):
    # List for storing predicted spam or not
    predictions = []

    # Iterte over legnth of emial data 
    for index in range(len(x_Test)):

        # Predictions of data
        predictions.append(knn_predict(x_Train, y_Train, x_Test[index], k))
    
    return predictions


def main():
    # Load data
    data = load_data()
    
    # Preprocess data, 75/25 split for test and train
    # x holds the data while y hold the last spam column
    x_Train, x_Test, y_Train, y_Test = preprocess_data(data)

    # Train the KNN model for spam prediction
    predictions = knn(x_Train, y_Train, x_Test, k=5)

    # Evaluate performance
    print("Performance Metrics:")

    accuracy = accuracy_score(y_Test, predictions)
    print("Accuracy:", accuracy)
    
    conf_Matrix = confusion_matrix(y_Test, predictions)
    tn, fp, fn, tp = conf_Matrix.ravel()
    print(f"False Positives: {fp}")
    print(f"True Positives: {tp}")

    auc = roc_auc_score(y_Test, predictions)
    print("AUC Score:", auc)

if __name__ == "__main__":
    main()