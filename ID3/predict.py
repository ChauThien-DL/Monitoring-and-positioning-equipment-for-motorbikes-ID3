import joblib
import pandas as pd
import time

model_path = 'decision_tree_model.pkl'
data_pre=[1,0,1,1,2,0]

def predict_new_data(input_data):
    loaded_model = joblib.load(model_path)

    column_names = ['Mode_anti-theft', 'Key', 'Vibration', 'Acceleration', 'Distance', 'Speed']
    input_df = pd.DataFrame([input_data], columns=column_names)


    prediction = loaded_model.predict(input_df)
    return prediction
while True:
    pre = predict_new_data(data_pre)
    print('Kết quả dự đoán: ',pre[0])
    time.sleep(10)
    

