import json

label_map = {0: 'sadness',
             1: 'anger',
             2: 'love',
             3: 'surprise',
             4: 'fear',
             5: 'joy'}

# Save label map
with open(r"C:\Users\kamal\OneDrive\Tài liệu\GitHub\NLP-Emotion-Analysis\models\label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f)
