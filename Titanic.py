train_features = []
test_features = []
train_labels = []
test_labels = []
features = [2,4,5,6,7,9]
features_t = [1,3,4,5,6,8]

import csv
import numpy as np

with open("train.csv") as csv_file:
    csv_reader = csv.reader(csv_file)
    cnt = 0
    for row in csv_reader:
        if cnt==0:
            cnt = cnt+1
            continue
        tmp_list = []
        #print row
        for k in features:
            if row[k]=="male":
                tmp_list.append(float('0'))
            elif row[k]=="female":
                tmp_list.append(float('1'))
            else:
                if row[k]=='':
                    value = 0
                else:
                    value=float(row[k])
                tmp_list.append(value)
        #print tmp_list
        train_features.append(np.array(tmp_list))
        train_labels.append(float(row[1]))

with open("test.csv") as csv_file:
    csv_reader = csv.reader(csv_file)
    cnt = 0
    for row in csv_reader:
        if row[0]=="PassengerId":
            continue
        tmp_list = []
        for k in features_t:
            if row[k]=="male":
                value = 0
            elif row[k]=="female":
                value = 1
            else:
                if row[k]=='':
                    value=0
                else:
                    value = row[k]
            tmp_list.append(float(value))
        test_features.append(np.array(tmp_list))
        test_labels.append(float(row[1]))

from sklearn.ensemble import RandomForestClassifier   
clf = RandomForestClassifier(min_samples_split=50, n_estimators=101)
clf.fit(train_features, train_labels)

pred = clf.predict(test_features)
print len(pred)

writer = csv.writer(open("result.csv", "wb"))
head = ["PassengerId", "Survived"]
writer.writerows([head])
cnt = 892
for row in pred:
    writer.writerows([[str(cnt), str(int(row))]])
    cnt = cnt+1