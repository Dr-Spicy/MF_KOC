import pandas as pd
import json
csv_file = "XHS_KOC_List - Sheet1.csv"
json_file = "processed/Ning/creator_creator_total.json"
df_csv = pd.read_csv(csv_file)
df_json = pd.read_json(json_file)
df_json = df_json.rename(columns={
    'nickname': 'Name', 
    'gender': 'Sex', 
    'desc': 'Bio(text)', 
    'follows': 'following count', 
    'fans': 'fans count', 
    'tag_list': 'tags(text)', 
    'interaction': 'liked n collected'  
})
for i in range(len(df_json)):
    rows = df_json.iloc[i]
    user_id = rows['user_id']
    csv_mask = df_csv['user_id'] == user_id
    for name in df_json.columns:
        if name == 'last_modify_ts' or name == 'avatar':
            continue
        elif name == 'tags(text)':
            df = pd.DataFrame([json.loads(rows['tags(text)'])])
            df_csv.loc[csv_mask, name] = str(df.dropna(axis=1).values[0])
        else:
            df_csv.loc[csv_mask, name] = rows[name]


output_csv_file = "XHS_KOC_List - transformed.csv"
df_csv.to_csv(output_csv_file, index=False, encoding="utf-8-sig")