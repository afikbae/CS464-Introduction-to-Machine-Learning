CS464 Introduction to Machine Learning
Fall 2024-25
Homework 1

Mehmet Akif Sahin
22203673

There is 3 python scripts provided:
	multinomial_naive_bayes.py
	bernoulli_naive_bayes.py
	q2main.py

You can run the python scripts using python3 with numpy installed.
Example:
	$ python3 q2main.py

This program will give output in following fashion:

Model Type
alpha = x
Accuracy: x.xxx
Confusion Matrix:
xx  xx  xx
xx  xx  xx
xx  xx  xx

Note that the multinomial naive bayes model will run twice, without a additive prior and with drichlet prior alpha = 1. you can change the alpha values in lines 20-24-28. 

You can also change the training and test datasets by inserting the paths of files to the constants TRAINING_FILES and TEST_FILES.
