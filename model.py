import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


X = [
[199, 216, 319, 267, 132], 
[321, 131, 285, 139, 154], 
[200, 113, 337, 159, 104],
[329, 252, 416, 182, 161],
[258, 159, 326, 269, 166],
[346, 322, 352, 313, 249], 
[251, 422, 400, 359, 238], 
[337, 416, 331, 250, 244],
[302, 334, 320, 206, 250],
[403, 420, 370, 354, 259],
]

b = [[224], [151], [205], [192], [232], [249], [238], [228], [216], [247]]
B = [232, 247, 111, 343, 282, 367, 334, 187, 325, 224]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X, b)

# The coefficients
print('Coefficients: \n', regr.coef_)


coef = [-0.02416354, -0.03407572, -0.05534695,  0.01777796,  0.05596865,
        -0.00080654,  0.04421302, -0.14429507, -0.10308929,  0.00317175,
         0.15165537,  0.02393767,  0.0527229 , -0.0249763 ,  0.18816539,
         0.08089365, -0.0580064 ,  0.04598031,  0.13215418,  0.04392422,
         0.00515431, -0.09893137,  0.11915368,  0.07632324, -0.0021731 ,
         0.09049858,  0.13892983,  0.06313879,  0.1310991 ]

print(len(list(coef)))
predictions = [0]*(len(X) ) 
for c in range(0, len(X) ):
	predictions[c] = np.dot(coef, X[c])

a = predictions
print("a = " + str(len(a)))
print("B = " + str(len(B)))

p = [0]*len(a)
for k in range (0, len(a)):
	p[k] = (a[k] - B[k]) * (a[k] - B[k])
print("Mean squared error: %.2f", np.mean(p) )
print('Variance score: %.2f' % regr.score(X, b))
