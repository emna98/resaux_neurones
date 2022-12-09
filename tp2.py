import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn import metrics


df = pd.read_csv('Iris.csv')
print(df.to_string()) 
print(df.head(10))
print("Taille du Dataframe: ",df.shape)

g = sns.pairplot(df,hue="Species", vars=["PetalLengthCm","PetalWidthCm"])
plt.show()

df['Species'] = df['Species'].replace(['Iris-setosa'], '0')
df['Species'] = df['Species'].replace(['Iris-versicolor'], '1')
df['Species'] = df['Species'].replace(['Iris-virginica'], '2')

print(df.head(10))

columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
x = df.loc[:, columns]
y = df.loc[:, ['Species']]
x_train, x_test, y_train, y_test = train_test_split(x,y.values.ravel(), test_size=0.3, random_state=5)
print("Données d'apprentissage:")
print(x_train.head(10))
print("\nDonnées de test:")
print(x_test.head(10))

#MLPClassifier avec taux d'apprentissage=0.7
clf = MLPClassifier(solver='lbfgs', max_iter=150, epsilon=0.07, learning_rate_init=0.7)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print("prediction: \n", y_predict)
print("accuracy = ",accuracy_score(y_test, y_predict))
matrice = confusion_matrix(y_test, y_predict)
print("\nMatrice de confusion: \n", matrice )

#Autre classifieur
model = MLPRegressor().fit(x_train, y_train.values.ravel())
print(model)
prediction= model.predict(x_test)
print( prediction)
print(metrics.r2_score(y_test, prediction))

#Etudier la variation du taux d'apprentissage
params= [ 
    {
        "solver":"sgd",
        "learning_rate":"constant",
        "learning_rate_init":0.2,
        "max_iter": 150,  
    },

    {
        "solver":"sgd",
        "learning_rate":"constant",
        "learning_rate_init":0.4,
        "max_iter": 150, 
    },
     {
        "solver":"sgd",
        "learning_rate":"invscaling",
        "learning_rate_init": 0.5,
        "max_iter": 300,
    },

     {
        "solver":"sgd",
        "learning_rate":"invscaling",
        "learning_rate_init": 0.7,
        "max_iter": 150,  
    }
]

labels=[
    "constant learning-rate_0.2",
    "constant learning-rate_0.4",
    "invscaling learning-rate_0.5",
    "invscaling learning-rate_0.7",
]

plot_args=[
    {"c": "red", "linestyle": "-"},
    {"c": "green", "linestyle": "-"},
    {"c": "blue", "linestyle": "-"},
    {"c": "red", "linestyle": "--"},
    {"c": "green", "linestyle": "--"},
]

mlps = []
for label, param in zip(labels, params):
    print('training : %s' % label)
    mlp=MLPClassifier(random_state=0, **param)
    mlp.fit(x_train, y_train)
    mlps.append(mlp)
    print("training set score : %f" % mlp.score(x_train,y_train))

for mlp,label,args in zip(mlps, labels, plot_args):
    plt.plot(mlp.loss_curve_)
    plt.title(" %s " %label, fontsize=12)
    plt.show()    
  

print("\n********************** iterations X 10 **************************\n")

clf = MLPClassifier(solver='lbfgs', max_iter=1500, epsilon=0.07, learning_rate_init=0.7)
clf.fit(x_train, y_train)
y_predict3 = clf.predict(x_test)
print("prediction: \n", y_predict3)
print("accuracy = ",accuracy_score(y_test, y_predict3))
matrice3 = confusion_matrix(y_test, y_predict3)
print("Matrice de confusion: \n", matrice3 )

