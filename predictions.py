import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

with open('anemia_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

new_data = pd.read_csv('new_data.csv')

label_encoder = LabelEncoder()
new_data['Gender'] = label_encoder.fit_transform(new_data['Gender'])
new_data['Past History (Individual)'] = label_encoder.fit_transform(new_data['Past History (Individual)'])
new_data['Any Symptoms'] = label_encoder.fit_transform(new_data['Any Symptoms'])
new_data['Food Preferences'] = label_encoder.fit_transform(new_data['Food Preferences'])
new_data['Past History (Immediate family)'] = label_encoder.fit_transform(new_data['Past History (Immediate family)'])

predictions = model.predict(new_data)

print("Predictions:")
for prediction in predictions:
    print(prediction)
