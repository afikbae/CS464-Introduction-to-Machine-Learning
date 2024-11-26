from bernoulli_naive_bayes import bernoulli_nb
from multinomial_naive_bayes import multinomial_nb

def printResults(name, alpha, accuracy, cm):
    print("------------------------------------------------------")
    print(name)
    print(f"alpha = {alpha}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Confusion Matrix:")
    for r in cm:
        for v in r:
            print(f"{int(v):4d}", end=' ')
        print()

TRAINING_FILES = ["../dataset/x_train.csv", "../dataset/y_train.csv"]
TEST_FILES = ["../dataset/x_test.csv", "../dataset/y_test.csv"]

classes = [0,1,2]

alpha = 0
mn_acc, mn_cm = multinomial_nb(TRAINING_FILES, TEST_FILES, classes, alpha)
printResults("Multinomial Naive Bayes Model", alpha, mn_acc, mn_cm)

alpha = 1
mn_acc, mn_cm = multinomial_nb(TRAINING_FILES, TEST_FILES, classes, alpha)
printResults("Multinomial Naive Bayes Model", alpha, mn_acc, mn_cm)

alpha = 1
b_acc, b_cm = bernoulli_nb(TRAINING_FILES, TEST_FILES, classes, alpha)
printResults("Bernoulli Naive Bayes Model", alpha, b_acc, b_cm)

print("------------------------------------------------------")
