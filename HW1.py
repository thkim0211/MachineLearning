import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

#Tae Hwan Kim
#I pledge my honor that I have abided by the Stevens Honor System.


iris = load_iris()

df= pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

X = df.iloc[0:150, [1, 2]].values
y = df.iloc[0:150, 5].values
# set output lable value to 1 if it is Virginca and 0 if Other.
y = np.where(y == 'virginica', 1, 0)

X_std = np.copy(X)
X_std[:,0] = (X_std[:,0] - X_std[:,0].mean()) / X_std[:,0].std()
X_std[:,1] = (X_std[:,1] - X_std[:,1].mean()) / X_std[:,1].std()
print(y)

X_train,X_test,y_train,y_test = train_test_split(X_std,y,test_size = 0.2)


def sigmoid(X, theta):
    z = np.dot(X, theta[1:]) + theta[0] 
    return 1.0 / ( 1.0 + np.exp(-z))

def lrCostFunction(y, hx):
  
    j = -y.dot(np.log(hx)) - ((1 - y).dot(np.log(1-hx)))
    
    return j


def error(X_std,theta,y): 
        hx = sigmoid(X_std,theta)
        c = lrCostFunction(y, hx)
        e = hx - y
        return e, c

def lrGradient(X_std, y, theta, alpha, num_iter):
    # empty list to store the value of the cost function over number of iterations
    cost = []
    
    for i in range(num_iter):
        e,c = error(X_std,theta,y)
        grad = X_std.T.dot(e)
        theta[0] = theta[0] - alpha * e.sum()
        theta[1:] = theta[1:] - alpha * grad
        
        cost.append(c)
        
    return cost,theta

theta = np.zeros(3)

alpha = 0.01
num_iter = 5000

cost,theta = lrGradient(X_train,y_train, theta, alpha, num_iter)

plt.plot(range(1, len(cost) + 1), cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')




print ('\n Logisitc Regression bias(intercept) term :', theta[0])
print ('\n Logisitc Regression estimated coefficients :', theta[1:])

plt.show()

def lrPredict(X_std,theta):
    
    return np.where(sigmoid(X_std,theta) >= 0.5,1, 0)


from matplotlib.colors import ListedColormap

def pdb(x,y,theta):
    ps = x
    label = y
    figure = plt.figure()
    graph = figure.add_subplot(1, 1, 1)
    x_a = []
    y_a = []
    for index, labelValue in enumerate(label):               
        pltx = ps[index][0]
        x_a.append(pltx)
        plty = ps[index][1]
        y_a.append(plty)
        if labelValue == 0:                   
            graph.scatter(pltx, plty, c='b', marker="x", label='X')
        else:
            graph.scatter(pltx, plty, c='r', marker="o", label='O')


pdb(X_test,y_test,theta)
plt.title('Decision Boundary')
plt.xlabel('sepal length ')
plt.ylabel('sepal width ')
plt.show()

def accuracy(x,theta,y):
    n_correct = 0
    m = len(x)
    pred = lrPredict(x,theta)
    for i in range(m):
        if y[i] == pred[i]:
            n_correct += 1
    print(f"Accuracy:{n_correct/len(y)}")

accuracy(X_test,theta,y_test)
accuracy(X_train,theta,y_train)
