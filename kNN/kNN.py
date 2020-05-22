import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

enc = preprocessing.LabelEncoder()

buying = enc.fit_transform(list(data["buying"]))
maint = enc.fit_transform(list(data["maint"]))
door = enc.fit_transform(list(data["door"]))
persons = enc.fit_transform(list(data["persons"]))
lug_boot = enc.fit_transform(list(data["lug_boot"]))
safety = enc.fit_transform(list(data["safety"]))
cls = enc.fit_transform(list(data["class"]))

X = list(zip(buying,maint,door,persons,lug_boot,safety))
Y = list(cls)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(X_train, Y_train)
acc = model.score(X_test, Y_test)
print(acc)

predicted = model.predict(X_test)
names = ["unacc", "acc", "good", "vgood"]

buying = ["vhigh", "high", "med", "low"]
maint = ["vhigh", "high", "med", "low"]
door = ["2", "3", "4", "5","more"]
persons = ["2", "4", "more"]
lug_boot = ["small", "med", "big"]
safety = ["low", "med", "high"]

names2 = np.array([buying,maint,door,persons,lug_boot,safety])


for x in range(len(predicted)):
    """names3 = "Price: " + buying[X_test[x][0]], "  Maintenence: " + maint[X_test[x][1]], "   Door: " + door[X_test[x][2]],\
             "  Persons: " + persons[X_test[x][3]], "   Lug Boot: " + lug_boot[X_test[x][4]], \
             "  Safety: " + safety[X_test[x][5]]"""

    print ("Data:"+ str(names3), "   Predicted: ", names[predicted[x]],  "   Actual: ", names[Y_test[x]])



"""print(X_test)
print(X_test[0][0])
print(names3)

names3 = "Price: "+ buying[X_test[0][0]], "  Maintenence: "+ maint[X_test[0][1]], "   Door: "+ door[X_test[0][2]],
         "  Persons: "+ persons[X_test[0][3]], "   Lug Boot: "+ lug_boot[X_test[0][4]], "  Safety: "+ safety[X_test[0][4]]"""

"""names2 = np.array([["vhigh", "high", "med", "low"],["vhigh", "high", "med", "low"],["2", "3", "4", "5more"],
                   ["2", "4", "more"],["small", "med", "big"],["low", "med", "high"]])"""

"""names2 = np.array([buying,maint,door,persons,lug_boot,safety])"""