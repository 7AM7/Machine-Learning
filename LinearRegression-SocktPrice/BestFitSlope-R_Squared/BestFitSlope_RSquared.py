from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

## create random  dataset for x and y 
def create_dataset(count, variance, step=2, correlation=False):
    val = 1 # first value initlaze
    ys = [] ## y value
    for i in range(count):
        y = val + random.randrange(-variance, variance) ## get random value for y in variance range
        ys.append(y)
        if correlation and correlation == 'pos': ## positve value
            val += step
        elif correlation and correlation == 'neg': ## negative value
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64),
## Y = m*x + b

## m and b 
def best_fit_slope_and_intercept(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
           ((mean(xs) * mean(xs)) - mean(xs*xs)) )
    b = mean(ys) - m*mean(xs)
    return m,b

## R squared

def squared_error(ys_orig, ys_line):  ## ys_line = regression_line = Y = m*x + b
    ## summation of regression_line - Y values all power of 2
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    ye_mean_line = [mean(ys_orig) for y in ys_orig] ## y_ (_ above y)
    squard_error_regr = squared_error(ys_orig, ys_line) ## SEy^ (^ above y)
    squard_error_y_mean = squared_error(ys_orig, ye_mean_line) ## SEy_ (_ above y)
    return 1 - (squard_error_regr / squard_error_y_mean)


## create random y and x value

xs, ys = create_dataset(40, 40, 2, correlation='pos') #form 0 to top like this /

#xs, ys = create_dataset(40, 40, 2, correlation='neg') #form top to 0 like this \

#xs, ys = create_dataset(40, 40, 2, correlation=False) # flat line like this --

# get m and b value
m, b = best_fit_slope_and_intercept(xs,ys)

regression_line = [(m*x)+b for x in xs]
##another way
# regression_line = []
#for x in xs:
 #   regression_line.append((m*x)+b)


 
## Here we can predict any values of y if we have a X values.
predict_x = 8
predict_y = (m*predict_x)+b



## R squared

## when R Squared near 0 we will have flat line 
r_squared = coefficient_of_determination(ys, regression_line)
print("R_Squared: ",r_squared)


## create graph for values x and y and draw regression line
plt.scatter(xs,ys) ##scatter =  add or throwing values of x and y on gprah
plt.scatter(predict_x,predict_y, s=100, color='g')
plt.plot(xs, regression_line)
plt.show()

