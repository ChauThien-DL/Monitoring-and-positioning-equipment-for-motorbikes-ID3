import pandas as pd

file_path = 'data.csv'  
data = pd.read_csv(file_path)

columns_to_check = ["Mode_anti-theft", "Key", "Vibration", "Acceleration", "Distance", "Speed", "Rules"]

for column in columns_to_check:
    if column in data.columns:  
        data = data[~data[column].astype(str).str.contains("Bỏ", na=False)]

def convert_text_to_number(value):
    mapping = {
        "Có": 1,
        "Mất": 1,
        "Mức 1": 1,
        "Mức 2": 2,
        "Không": 0
    }
    return mapping.get(value, value)  
columns_to_convert = ["Mode_anti-theft", "Key", "Vibration", "Acceleration", "Distance", "Speed", "Rules"]

for column in columns_to_convert:
    if column in data.columns:  
        data[column] = data[column].apply(convert_text_to_number)
        
output_file = 'hihi.csv'
data.to_csv(output_file, index=False)
print(f"File đã được lưu sau khi chuyển đổi tại: {output_file}")
