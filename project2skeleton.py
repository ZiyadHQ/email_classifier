from collections import defaultdict
import glob
import math
import os

class Model:

    def __init__(self):
        self.ham_count = 0
        self.ham_fd = defaultdict(int)

        self.spam_count = 0
        self.spam_fd = defaultdict(int)

def load_data(directory):
    x = []
    y = []
    for f in glob.glob(os.path.join(directory,"HAM.*.txt")):
        with open( f, 'r')as file:
            x.append(file.read())
            y.append(0)
    for f in glob.glob(os.path.join(directory,"SPAM.*.txt")):
        with open( f, 'r')as file:
            x.append(file.read())
            y.append(1)
    return x,y

def nb_train(x, y):
    model = Model()
    for email, label in zip(x, y):
        if label == 0: model.ham_count += 1
        else: model.spam_count += 1

        for tok in email.split(' '):
            if label == 0: model.ham_fd[tok] += 1
            else: model.spam_fd[tok] += 1
    
    return model

def safe_log(value, fallback = -float('inf')):
    
    if value <= 0:
        return fallback
    else:
        return math.log(value)

def calculate_probability(email, word_probs, total_count, vocab_size, use_log=False, smoothing=1):
    prob = 0 if use_log else 1
    for word in email.split():
        word_prob = (word_probs.get(word, 0) + smoothing) / (total_count + smoothing * vocab_size)
        if use_log:
            prob += safe_log(word_prob)
        else:
            prob *= word_prob
    
    return prob

def nb_test(docs, trained_model, use_log = False, smoothing = False):
    pred = []

    vocab_size = len(set(trained_model.ham_fd.keys()).union(set(trained_model.spam_fd.keys())))

    total_ham = sum(trained_model.ham_fd.values())
    total_spam = sum(trained_model.spam_fd.values())

    for email in docs:
        ham_prop = calculate_probability(email, trained_model.ham_fd, total_ham, vocab_size, use_log, smoothing)
        spam_prop = calculate_probability(email, trained_model.spam_fd, total_spam, vocab_size, use_log, smoothing)

        pred.append(0 if ham_prop > spam_prop else 1)
    
    return pred

def f_score(y_true, y_pred):
    
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for real, pred in zip(y_true ,y_pred):
        if real == 1 and pred == 1:
            true_positive += 1
        elif real == 0 and pred == 1:
            false_positive += 1
        elif real == 1 and pred == 0:
            false_negative += 1
    
    precision = true_positive/(true_positive + false_positive)

    recall = true_positive/(true_positive + false_negative)

    f = 2 * ((precision * recall) / (precision + recall))

    return f, precision, recall

x_train, y_train = load_data("./SPAM_training_set/")
model = nb_train(x_train, y_train)

x_test, y_test = load_data("./SPAM_test_set/")
y_pred = nb_test(x_test, model, use_log = False, smoothing = False)

print("without using log, without using smoothing: ", f_score(y_test,y_pred))

y_pred = nb_test(x_test, model, use_log = False, smoothing = True)

print("without using log, with smoothing: ", f_score(y_test,y_pred))

y_pred = nb_test(x_test, model, use_log = True, smoothing = False)

print("with log, without using smoothing: ", f_score(y_test,y_pred))

y_pred = nb_test(x_test, model, use_log = True, smoothing = True)

print("with log, with smoothing: ", f_score(y_test,y_pred))
