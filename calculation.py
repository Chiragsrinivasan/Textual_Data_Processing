import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

data = pd.read_csv('anemia_dataset_with_result.csv')
print(data.head())

label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Past History (Individual)'] = label_encoder.fit_transform(data['Past History (Individual)'])
data['Any Symptoms'] = label_encoder.fit_transform(data['Any Symptoms'])
data['Food Preferences'] = label_encoder.fit_transform(data['Food Preferences'])
data['Past History (Immediate family)'] = label_encoder.fit_transform(data['Past History (Immediate family)'])

print(data.head())

X = data.drop(['Anemia (individual)', 'Result'], axis=1)
y = data['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv('train_anemia_dataset.csv', index=False)
X_test.to_csv('test_anemia_dataset.csv', index=False)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'anemia_detection_model.pkl')

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
