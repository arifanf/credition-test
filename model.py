import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

credit = pd.read_csv('credit_cleaned.csv')

df = credit.copy()
target = ['JENIS PENJUALAN STNK']
encode = ['TYPE MOTOR', 'COLOR', 'JENIS KELAMIN', 'STATUS RUMAH', 'PEKERJAAN',
          'MERK MOTOR SBLMNYA', 'TYPE MOTOR SBLMNYA', 'SMH DIGUNAKAN UNTUK', 'YG MENGGUNAKAN SMH', 'HOBI']

for col in encode:    
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'CASH':0, 'CREDIT':1}
def target_encode(val):
    return target_mapper[val]

df['JENIS PENJUALAN STNK'] = df['JENIS PENJUALAN STNK'].apply(target_encode)

# Separating X and y
X = df.drop('JENIS PENJUALAN STNK', axis=1)
y = df['JENIS PENJUALAN STNK']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Build random forest model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Saving the model
pickle.dump(clf, open('credit.pkl', 'wb'))