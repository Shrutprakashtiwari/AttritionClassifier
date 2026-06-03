import pandas as pd
df=pd.read_csv('/content/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
# df.isnull().sum()x = x.drop("EmployeeNumber", axis=1)
x=df.drop(['Attrition','EmployeeNumber'],axis=1)

y=df['Attrition']
x = df.drop('Attrition', axis=1)

x = pd.get_dummies(x, drop_first=True)

y = LabelEncoder().fit_transform(df['Attrition'])
# x.head()
# y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100, class_weight="balanced",max_depth=8,
    min_samples_leaf=5,
    random_state=42)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))

importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": rf.feature_importances_
})

print(
    importance.sort_values(
        by="Importance",
        ascending=False
    ).head(10)
)
from sklearn.metrics import recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

c_values = [0.001, 0.01, 0.1, 1, 10, 100]

recalls = []
f1s = []

# for c in c_values:

#     lr = LogisticRegression(
#         class_weight='balanced',
#         C=c,
#         random_state=42
#     )

#     lr.fit(x_train, y_train)

#     y_pred = lr.predict(x_test)

#     recalls.append(recall_score(y_test, y_pred))
#     f1s.append(f1_score(y_test, y_pred))

# plt.plot(c_values, recalls, marker='o', label='Recall')
# plt.plot(c_values, f1s, marker='o', label='F1')

# plt.xscale('log')
# plt.xlabel('C')
# plt.ylabel('Score')
# plt.legend()
# plt.show()

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
pipe=Pipeline([
    ('classifier',LogisticRegression(class_weight='balanced',C=.001)),
])
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))

