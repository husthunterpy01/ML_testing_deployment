import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# File loading
df = pd.read_csv("iris.csv")
print(df.head())

X = df[["Sepal_Length","Sepal_Width","Petal_Length","Petal_Width"]]
y = df["Class"]

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=45)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier()

classifier.fit(X_train,y_train)

# Save the model
pickle.dump(classifier, open("model.pkl","wb"))
