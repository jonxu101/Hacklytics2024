import pandas as pd

df = pd.read_csv('predicted.csv')
all_data = pd.read_csv('study_data_test_set.csv')

# print(all_data)

n = len(df)
df.insert(len(df.columns), "Website", all_data['Study URL'][:n])
df.insert(len(df.columns), "NCT", all_data['NCT Number'][:n])
df.insert(len(df.columns), "Condition", all_data['Conditions'][:n])
# df.insert(len(df.columns), "OtherInfo", ["cancer" for i in range(len(df))])

df.to_csv('predicted_out.csv', index=False)
