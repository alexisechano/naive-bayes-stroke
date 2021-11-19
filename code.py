# Alexis Echano
# CS 109 Challenge Code
# November 2021

# import statements
import csv
import math
import random

# features used and their corresponding column heading for cleaner variable usage
ID = 'id' # unique identifier
GENDER = 'gender' # 1 for female, 0 for male
AGE = 'age' # 1 for older than or equal to 65 (stroke science), 0 for younger
HYPERTENSION = 'hypertension' # 1 for yes, 0 for no
HEART_DISEASE = 'heart_disease' # 1 for yes, 0 for no
MARRIED = 'ever_married' # 1 for yes, 0 for no
GLUCOSE_LEVEL = 'avg_glucose_level' # 1 for above 108 (hyperglycemia), 0 for below
SMOKED = 'smoking_status' # 1 for formerly or current, 0 for others
STROKE = 'stroke' # 1 for yes, 0 for no

# cleaned column headings and indices
cleaned_features = [ID, GENDER, AGE, HYPERTENSION, HEART_DISEASE, MARRIED, GLUCOSE_LEVEL, SMOKED]
IDX_ID = 0
IDX_GENDER = 1
IDX_AGE = 2
IDX_HYPERTENSION = 3
IDX_HEART_DISEASE = 4
IDX_MARRIED = 5
IDX_GLUCOSE_LEVEL = 6
IDX_SMOKED = 7
IDX_STROKE = 8

# constants - determined by research articles in writeup
AGE_OF_INCREASED_RISK = 65
GLUCOSE_LEVEL_INCREASED_RISK = 108.0

# ------------------------------- PREPROCESSING CODE -------------------------------- #

# input is string row from CSV file, using binarize()
def clean_row(row):
    clean = []

    # iterate through row values, clean should have all 0 and 1 except for ID
    clean.append(row[0]) # id, stays as string
    clean.append(binarize(row[1], GENDER)) # gender
    clean.append(binarize(math.ceil(float(row[2])), AGE)) # age
    clean.append(int(row[3])) # hypertension
    clean.append(int(row[4])) # heart disease
    clean.append(binarize(row[5], MARRIED)) # ever married
    clean.append(binarize(row[8], GLUCOSE_LEVEL)) # avg glucose level
    clean.append(binarize(row[10], SMOKED)) # smoking status
    clean.append(int(row[11])) # stroke

    # return cleaned list
    return clean

# function to enable us to use Bernoulli Naive Bayes by binarizing given data point
# returns -1 if nothing can be done
def binarize(dataval, type):
    binarized_val = -1
    if type == GENDER:
        binarized_val = 1 if dataval == 'female' else 0
    elif type == AGE:
        binarized_val = 1 if int(dataval) >= AGE_OF_INCREASED_RISK else 0
    elif type == MARRIED:
        binarized_val = 1 if dataval == 'Yes' else 0
    elif type == GLUCOSE_LEVEL:
        binarized_val = 1 if float(dataval) == GLUCOSE_LEVEL_INCREASED_RISK else 0
    elif type == SMOKED:
        binarized_val = 1 if dataval == 'formerly smoked' or dataval == 'smokes' else 0
    return binarized_val

# reads in CSV file and cleans the rows
def read_and_load_data():
    total_cleaned_data = []
    with open('stroke_data.csv', mode='r') as stroke_file:
        # iterate through data rows
        csv_reader = csv.reader(stroke_file, delimiter=',')
        next(csv_reader) # skip first row of column names
        
        for row in csv_reader:
            cleaned_row = clean_row(row)
            total_cleaned_data.append(cleaned_row)
    
    # if data is properly cleaned and stored...
    if len(total_cleaned_data) != 0:

        # split into random training and testing groups, 2/3 testing, 1/3 training
        train_size = math.ceil((1 * len(total_cleaned_data))/3)

        # shuffle (randomized) and split the data
        random.shuffle(total_cleaned_data)
        total_training_data = total_cleaned_data[0:train_size]
        total_testing_data = total_cleaned_data[train_size:]
        return total_training_data, total_testing_data
    
    return -1

# ------------------------------- TRAINING CODE -------------------------------- #

# probability of priors or class probabilities, no matter what feature
def calc_priors(total_training_data):
    total = len(total_training_data)
    count_stroke = 0
    count_no_stroke = 0

    for patient in total_training_data:
        if patient[IDX_STROKE] == 1:
            count_stroke += 1
        else:
            count_no_stroke += 1
    
    # calculate simple stroke probabilities with Laplace
    P_stroke = float(count_stroke + 1)/(total + 2)
    P_no_stroke = float(count_no_stroke + 1)/(total + 2)

    return P_stroke, P_no_stroke
    
# probability of stroke, given features
# return is list of lists: [P(X = 1 | stroke), P(X = 0 | stroke), P(X = 1 | no stroke), P(X = 0 | no stroke)]
def calc_likelihoods(training_data):
    probabilities = [] 
    for feature in range(len(cleaned_features)):
        count_1_given_stroke = 0
        count_0_given_stroke = 0
        total_feature_count_stroke = 0
        for t in training_data:
            curr_val = t[feature]
            curr_stroke = t[IDX_STROKE]

            if curr_stroke == 1:
                if curr_val == 1:
                    count_1_given_stroke += 1
                else:
                    count_0_given_stroke += 1
                total_feature_count_stroke += 1
        
        p_1_given_stroke =  (float(count_1_given_stroke) + 1.0)/(total_feature_count_stroke + 2)
        p_0_given_stroke =  (float(count_0_given_stroke) + 1.0)/(total_feature_count_stroke + 2)
        probabilities.append([p_1_given_stroke, p_0_given_stroke, 1.0 - p_1_given_stroke, 1.0 - p_0_given_stroke])

    return probabilities

# multiply priors by conditionals per feature
def calc_product(probs, prior_stroke, prior_no_stroke):
    # probs input: [P(X = 1 | stroke), P(X = 0 | stroke), P(X = 1 | no stroke), P(X = 0 | no stroke)]
    products = []

    # calculate the probabilities of features occuring given stroke status
    for p in probs:
        p_1_given_stroke = prior_stroke * p[0]
        p_0_given_stroke = prior_stroke * p[1]
        p_1_given_no_stroke = prior_no_stroke * p[2]
        p_0_given_no_stroke = prior_no_stroke * p[3]
        products.append([p_1_given_stroke, p_0_given_stroke, p_1_given_no_stroke, p_0_given_no_stroke])
    return products

# train "model" to use for classification
def train(data):
    # calc class probabilities or priors
    P_stroke, P_no_stroke = calc_priors(data)

    # calculate conditional probabilities
    probabilities = calc_likelihoods(data)

    # calc initial products
    prods = calc_product(probabilities, P_stroke, P_no_stroke)

    return prods
    
# ------------------------------- CLASSIFICATION CODE -------------------------------- #

# calculate simple probabilties of features showing up or not, matches cleaned_features, using Laplace
def calc_feature_evidence(training_data):
    probabilities = [] # lists are [P(X = 0), P(X = 1)]
    n = len(training_data)
    for feature in range(len(cleaned_features)):
        curr_feature_count_1 = 0
        curr_feature_count_0 = 0
        for t in training_data:
            if t[feature] == 1:
                curr_feature_count_1 += 1
            else:
                curr_feature_count_0 += 1
        
        # append probabilties to list
        p_1 = (curr_feature_count_1 + 1)/(n + 2)
        p_0 = (curr_feature_count_0 + 1)/(n + 2)
        probabilities.append([p_0, p_1])
    return probabilities

def classify(training_data, testing_data, products):
    # determine individual feature class probs
    feature_class_probs = calc_feature_evidence(training_data)

    successes = 0
    n = len(testing_data)
    for t in testing_data:
        feature_vals = t[0:len(t) - 1]
        actual_value = t[len(t) - 1]

        # multiple evidence for features (not given anything)
        p_evidence = 1.0
        p_cond_stroke = 1.0
        p_cond_no_stroke = 1.0
        for feature in range(len(feature_vals)):
            val = feature_vals[feature]
            if val == 1:
                p_evidence *= feature_class_probs[feature][1]
                p_cond_stroke *= products[feature][0]
                p_cond_no_stroke *= products[feature][2]
            else:
                p_evidence *= feature_class_probs[feature][0]
                p_cond_stroke *= products[feature][1]
                p_cond_no_stroke *= products[feature][3]
        
        # calculate conditional probabilities for both stroke and no stroke
        P_stroke = p_cond_stroke/p_evidence
        P_no_stroke = p_cond_no_stroke/p_evidence

        # determine stroke outcome
        exp_value = -1
        if P_stroke > P_no_stroke:
            exp_value = 1
        else:
            exp_value = 0
        
        # check success
        if exp_value == actual_value:
            successes += 1
    
    # determine how accurate the model is at classifying
    accuracy = successes/n

    return accuracy

def run():
    # process and train data
    training_data, testing_data = read_and_load_data()
    products = train(training_data)

    # ensure training data is classified well
    accuracy_train = classify(training_data, training_data, products)

    # using testing data
    accuracy_test = classify(training_data, testing_data, products)

    return accuracy_train, accuracy_test

# ------------------------------- MAIN METHOD CODE -------------------------------- #
def main():
    print("Starting up predictor/classifier...")
    print(" ")

    # run random samples of entire dataset N times
    n = 100
    sum_train = 0.0
    sum_test = 0.0
    for i in range(n):
        accuracy_train, accuracy_test = run()
        sum_train += accuracy_train
        sum_test += accuracy_test
    
    print("Average training dataset (1/3 of original) accuracy: ")
    print(sum_train/n)
    print(" ")
    print("Average testing dataset (2/3 of original) accuracy: ")
    print(sum_test/n)
    
    return None

if __name__ == "__main__":
    main()