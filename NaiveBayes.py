import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


    
def load_data():

    # Load your data from Excel or any other source
    data = pd.read_excel('Documents\Spam_finder\spambase.xlsx')

    # Shuffle the row indices
    shuf_Indices = np.random.permutation(data.index)

    # Use the shuffled indices to reorder the rows of the DataFrame
    data_Shuf = data.iloc[shuf_Indices]

    #split our data into 75% testing and 25% for testing
    test_Size = len(data_Shuf)//4
    testing_Indicies = (0, test_Size-1)
    train_Indicies = (test_Size, len(data_Shuf)-1)

    train_Data = data_Shuf.iloc[train_Indicies[0]:train_Indicies[1]+1]
    test_Data = data_Shuf.iloc[testing_Indicies[0]:testing_Indicies[1]+1]

    # splitting the remaining 3/4ths for 5 fold verification
    fold_Indices = []
    folds = 5
    fold_Size = len(train_Data) // folds

    # find start and stop indicies for each cross validation
    for current_Fold in range(folds):
        fold_Start = current_Fold*fold_Size
        if current_Fold < folds-1:
            fold_End = (current_Fold+1)*fold_Size -1
        else:
            fold_End = (current_Fold+1)*fold_Size
        fold_Indices.append((fold_Start, fold_End))
    
    #return sorted data for training
    # train_set = 3/4ths of data not used for testing
    # testing_set = indieces of data for testing training 
    # fold_indices = 5 sets of indicies for training 
    return train_Data, test_Data, fold_Indices


def train_naive_bayes(training_Data, five_Fold_Indicies):
    # Init empty lost for each fold probabilties 
    fold_Probabilities = []

    # Loop each fold for training on data
    for fold_Start, fold_End in five_Fold_Indicies:

        # Remove the data that is not currently being used for training
        fold_Data = training_Data.drop(training_Data.index[fold_Start:fold_End + 1])

        # Calc likelihood of each attribute for spam and not spam
        fold_Probability = calculate_likelihood(fold_Data)

        # Store the found fold probabilites
        fold_Probabilities.append(fold_Probability)
    
    # Average the total vlaue for each spam and non spam attribute of the 5 folds
    averaged_FiveFold_Probability = average_probabilities(fold_Probabilities)

    return averaged_FiveFold_Probability


def calculate_likelihood(training_Data):

    # count amount of emials in data
    num_Emails = len(training_Data)

    # dictionary for holding column names and probability of features spam or email
    probability = {}

    #create dictionary for storing processed data
    for column in training_Data.columns:
        if column != 'spam':
            # Create counters for emails
            spam_Counter = 0
            email_Counter = 0

            #populate the created dictionary
            for index, value in training_Data[column].items():

                # check if value is greator than 0
                if value > 0:

                    # incirment based on spam classification
                    if training_Data.loc[index, 'spam'] == 1:
                        spam_Counter += 1
                    else:
                        email_Counter += 1

            # calcualte the prability for spam and non spam in the current attribute
            probability[column] = {'spam': spam_Counter/num_Emails, 'email': email_Counter/num_Emails}

    return probability     

def average_probabilities(probabilities_List):
    # Initialize dictionary to hold averaged probabilities
    averaged_Probability = {}

    # Iterate over each feature
    for feature in probabilities_List[0].keys():

        # Initialize counters for spam and email probabilities
        spam_Total = 0
        email_Total = 0

        # Calculate the sum of probabilities for the current feature across all folds
        for fold_Probability in probabilities_List:
            spam_Total += fold_Probability[feature]['spam']
            email_Total += fold_Probability[feature]['email']

        # Calculate the average probabilities for the current feature
        averaged_Spam_Prob = spam_Total / len(probabilities_List)
        averaged_Email_Prob = email_Total / len(probabilities_List)

        # Store the averaged probabilities in the dictionary
        averaged_Probability[feature] = {'spam': averaged_Spam_Prob, 'email': averaged_Email_Prob}

    return averaged_Probability
                

def naive_predictor(probabilities, testing_Data):

    # list for predictions
    predictions = []

    for index, email in testing_Data.iterrows():
        spam_Probability = 1
        non_Spam_Probability = 1

        # calcualte the spam and non spam probs in current attribute
        for feature, value in email.items():
            if feature != 'spam':
                spam_Probability *= probabilities[feature]['spam'] if value else (1 - probabilities[feature]['spam'])
                non_Spam_Probability *= probabilities[feature]['email'] if value else (1 - probabilities[feature]['email'])

        # Compare probabilities and predict spam or not
        if spam_Probability > non_Spam_Probability:
            predictions.append(1)  # Predicted as spam
        else:
            predictions.append(0)  # Predicted as not spam

    return predictions
 

def main():
    # Load training and testing data
    training_Data, testing_Data, fiveFold_Indicies = load_data()

    # Train the Naive Bayes classifier
    #prior_spam, prior_not_spam, likelihood_spam, likelihood_not_spam = train_naive_bayes(training_data)
    data_Probability = train_naive_bayes(training_Data, fiveFold_Indicies)

    # Test the Naive Bayes classifier
    predictions = naive_predictor(data_Probability, testing_Data)

    actual_Labels = testing_Data['spam'].tolist()

    # Calculate accuracy using sklearn helper
    accuracy = accuracy_score(actual_Labels, predictions)
    print("Accuracy:", accuracy)

    # Calculate True Positive and False Positive using confusion matrix
    tn, fp, fn, tp = confusion_matrix(actual_Labels, predictions).ravel()
    print("True Positive:", tp)
    print("False Positive:", fp)

    # Calculate AUC
    auc = roc_auc_score(actual_Labels, predictions)
    print("AUC:", auc)

if __name__ == "__main__":
    main()