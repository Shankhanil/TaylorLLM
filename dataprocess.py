from glob import glob
import pandas as pd
import os

columns = ['file']
df = pd.DataFrame(columns = columns)

print(df)
path = os.path.join('data/Albums', '*')
# print(glob(path))
# exit()
i = 0
for folder in glob(path):

    filepath = os.path.join(folder, '*.txt')

    for file in glob(filepath):
        # with open(file, 'r') as fp:
        #     text = fp.read()

        df.loc[i] = {'file': file}
        i+= 1
print(df.head())
df.to_csv('dump.csv')
        # print(text)
