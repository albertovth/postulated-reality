
# This is a quick exercise to illustrate how a decision tree classifier can be trained to predict grade distributions.
# The starting point is a hypothetical situation, where a group of twenty examiners are asked to grade fifty papers, assigning a grade to each paper, in the scale A-F.
# Each of the examiners belongs to a particular group, out of four possible ones. The hypothesis is that each of the groups grades differently.
# Therefore, each examiner is presented as one of these four possible groups, which make out the categorical data.
# The decision tree can contribute to illustrate when a grade distribution varies from the expected distribution, according to a particular group.
# The groups have been one-hot encoded previously, to make the calculations easier.
# One-hot encoding could have been included in this exercise, but it would make it longer, and there would be less focus on the decision tree part.

# Model and prediction for inexperienced internal examiner who has taken an examination course

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Data has been generated randomly within previously defined intervals

Y = [[7,10,17,10,4,2],[8,12,13,10,4,3],[11,12,13,9,3,3],[6,12,15,10,4,3],[8,11,16,9,4,4],[10,12,13,9,4,3],[9,11,17,11,1,3],[9,12,13,10,3,3],[10,10,15,9,4,2],[9,11,13,10,4,3],[8,11,16,10,1,4],[10,9,15,9,4,3],[10,8,16,8,4,3],[10,11,15,10,2,2],[8,13,13,10,2,4],[10,12,14,9,2,3],[11,10,16,9,1,3],[11,11,14,10,1,3],[9,10,16,8,3,3],[10,9,16,9,3,3]]

X = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,0,1],[0,0,1,0],[1,0,0,0]]


Labels = ['A', 'B', 'C', 'D', 'E', 'F']
clf=DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction_interne_med_sensor_kurs = clf.predict([[1,0,0,0]])

print(prediction_interne_med_sensor_kurs)

Pred_M = pd.DataFrame(prediction_interne_med_sensor_kurs)
Pred_M.columns = Labels

print('Prediction = Grade distribution for an inexperienced internal examiner who has taken an examination course ' + str(Pred_M.iloc[0]))

Y_predict = clf.predict(X)
import numpy as np
y_true = np.array(Y)
y_pred = np.array(Y_predict)

hamming_loss=np.sum(np.not_equal(y_true, y_pred))/float(y_true.size)

print(np.sum(np.equal(y_true, y_pred)))

print('Hamming loss= '+ str(hamming_loss))

accuracy_score=np.sum(np.equal(y_true, y_pred))/float(y_true.size)

print('Accuracy score= '+ str(accuracy_score))

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

objects = Labels
y_pos = np.arange(len(objects))
performance_m = Pred_M.iloc[0]

plt.bar(y_pos, performance_m, align = 'center', alpha=0.5)
plt.xticks(y_pos, Labels, rotation='horizontal')
plt.ylabel('Frequency')
plt.title('Prediction = Grade distribution for an inexperienced internal examiner with an examination course')

plt.show()



# Model and prediction for inexperienced internal examiner who has not taken an examination course

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Data has been generated randomly within previously defined intervals
# This is the same data. It is presented again just to make it easier to see how it all fits together.

Y = [[7,10,17,10,4,2],[8,12,13,10,4,3],[11,12,13,9,3,3],[6,12,15,10,4,3],[8,11,16,9,4,4],[10,12,13,9,4,3],[9,11,17,11,1,3],[9,12,13,10,3,3],[10,10,15,9,4,2],[9,11,13,10,4,3],[8,11,16,10,1,4],[10,9,15,9,4,3],[10,8,16,8,4,3],[10,11,15,10,2,2],[8,13,13,10,2,4],[10,12,14,9,2,3],[11,10,16,9,1,3],[11,11,14,10,1,3],[9,10,16,8,3,3],[10,9,16,9,3,3]]

X = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,0,1],[0,0,1,0],[1,0,0,0]]

clf=DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction_interne_uten_sensor_kurs = clf.predict([[0,1,0,0]])

print(prediction_interne_uten_sensor_kurs)

Pred_U = pd.DataFrame(prediction_interne_uten_sensor_kurs)
Pred_U.columns = Labels

print('Prediction = Grade distribution for an inexperienced internal examiner who has not taken an examination course' + str(Pred_U.iloc[0]))

Y_predict = clf.predict(X)
import numpy as np
y_true = np.array(Y)
y_pred = np.array(Y_predict)
hamming_loss=np.sum(np.not_equal(y_true, y_pred))/float(y_true.size)

print('Hamming loss= '+ str(hamming_loss))

accuracy_score=np.sum(np.equal(y_true, y_pred))/float(y_true.size)

print('Accuracy score= '+ str(accuracy_score))

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

objects = Labels
y_pos = np.arange(len(objects))
performance_u = Pred_U.iloc[0]

plt.bar(y_pos, performance_u, align = 'center', alpha=0.5)
plt.xticks(y_pos, Labels, rotation='horizontal')
plt.ylabel('Frequency')
plt.title('Prediction = Grade distribution for an inexperienced internal examiner without examination course')

plt.show()


# Model and prediction for experienced internal examiner

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Data has been generated randomly within previously defined intervals
# This is the same data. It is presented again just to make it easier to see how it all fits together.

Y = [[7,10,17,10,4,2],[8,12,13,10,4,3],[11,12,13,9,3,3],[6,12,15,10,4,3],[8,11,16,9,4,4],[10,12,13,9,4,3],[9,11,17,11,1,3],[9,12,13,10,3,3],[10,10,15,9,4,2],[9,11,13,10,4,3],[8,11,16,10,1,4],[10,9,15,9,4,3],[10,8,16,8,4,3],[10,11,15,10,2,2],[8,13,13,10,2,4],[10,12,14,9,2,3],[11,10,16,9,1,3],[11,11,14,10,1,3],[9,10,16,8,3,3],[10,9,16,9,3,3]]

X = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,0,1],[0,0,1,0],[1,0,0,0]]


clf=DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction_interne_med_erfaring = clf.predict([[0,0,1,0]])

print(prediction_interne_med_erfaring)

Pred_V = pd.DataFrame(prediction_interne_med_erfaring)
Pred_V.columns = Labels

print('Prediction = Grade distribution for experienced internal examiner ' + str(Pred_V.iloc[0]))

Y_predict = clf.predict(X)
import numpy as np
y_true = np.array(Y)
y_pred = np.array(Y_predict)
hamming_loss=np.sum(np.not_equal(y_true, y_pred))/float(y_true.size)

print('Hamming loss= '+ str(hamming_loss))

accuracy_score=np.sum(np.equal(y_true, y_pred))/float(y_true.size)

print('Accuracy score= '+ str(accuracy_score))

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

objects = Labels
y_pos = np.arange(len(objects))
performance_v = Pred_V.iloc[0]

plt.bar(y_pos, performance_v, align = 'center', alpha=0.5)
plt.xticks(y_pos, Labels, rotation='horizontal')
plt.ylabel('Frequency')
plt.title('Prediction = Grade distribution for an experienced internal examiner')

plt.show()


# Model and prediction for external examiner

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Data has been generated randomly within previously defined intervals
# This is the same data. It is presented again just to make it easier to see how it all fits together.

Y = [[7,10,17,10,4,2],[8,12,13,10,4,3],[11,12,13,9,3,3],[6,12,15,10,4,3],[8,11,16,9,4,4],[10,12,13,9,4,3],[9,11,17,11,1,3],[9,12,13,10,3,3],[10,10,15,9,4,2],[9,11,13,10,4,3],[8,11,16,10,1,4],[10,9,15,9,4,3],[10,8,16,8,4,3],[10,11,15,10,2,2],[8,13,13,10,2,4],[10,12,14,9,2,3],[11,10,16,9,1,3],[11,11,14,10,1,3],[9,10,16,8,3,3],[10,9,16,9,3,3]]

X = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,0,1],[0,0,1,0],[1,0,0,0]]

clf=DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction_eksterne = clf.predict([[0,0,0,1]])

print(prediction_eksterne)

Pred_E = pd.DataFrame(prediction_eksterne)
Pred_E.columns = Labels

print('Prediction = Grade distribution for external examiner ' + str(Pred_E.iloc[0]))

Y_predict = clf.predict(X)
import numpy as np
y_true = np.array(Y)
y_pred = np.array(Y_predict)
hamming_loss=np.sum(np.not_equal(y_true, y_pred))/float(y_true.size)

print('Hamming loss= '+ str(hamming_loss))

accuracy_score=np.sum(np.equal(y_true, y_pred))/float(y_true.size)

print('Accuracy score= '+ str(accuracy_score))

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

objects = Labels
y_pos = np.arange(len(objects))
performance_e = Pred_E.iloc[0]

plt.bar(y_pos, performance_e, align = 'center', alpha=0.5)
plt.xticks(y_pos, Labels, rotation='horizontal')
plt.ylabel('Frekvens')
plt.title('Prediction = Grade distribution for an external examiner')

plt.show()


# Line diagram to illustrate the distributions


import matplotlib.pyplot as plt

plt.style.use('ggplot')

plt.plot(Labels, performance_m, '-X',

         label='Prediction: Grade distribution for inexperienced internal examiner with examination course')

plt.plot(Labels, performance_u, '-X',

         label='Prediction: Grade distribution for inexperienced internal examiner without examination course')

plt.plot(Labels, performance_v, '-X',

         label='Prediction: Grade distribution for experienced internal examiner')

plt.plot(Labels, performance_e, '-X',

         label='Prediction: Grade distribution for external examiner')

plt.legend(loc='right', bbox_to_anchor=(1,1))

plt.show()

# Model and prediction for an examination committee consisting of two examiners

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Data has been generated randomly within previously defined intervals
# This is the same data. It is presented again just to make it easier to see how it all fits together.

Y = [[7,10,17,10,4,2],[8,12,13,10,4,3],[11,12,13,9,3,3],[6,12,15,10,4,3],[8,11,16,9,4,4],[10,12,13,9,4,3],[9,11,17,11,1,3],[9,12,13,10,3,3],[10,10,15,9,4,2],[9,11,13,10,4,3],[8,11,16,10,1,4],[10,9,15,9,4,3],[10,8,16,8,4,3],[10,11,15,10,2,2],[8,13,13,10,2,4],[10,12,14,9,2,3],[11,10,16,9,1,3],[11,11,14,10,1,3],[9,10,16,8,3,3],[10,9,16,9,3,3]]

X = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,0,1],[0,0,1,0],[1,0,0,0]]

Sensorgruppe = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
Type_sensorgruppe = ['Internal and inexperienced with examination course', 'Internal and inexperienced without examination course', 'Internal and experienced', 'External']

Sensorgrupper_og_navn = pd.DataFrame(
    {'Sensorgruppe': Sensorgruppe,
     'Type sensor': Type_sensorgruppe,
    })
print(Sensorgrupper_og_navn)

sensor_1 = input("Please register the type for examiner 1. This can be Internal and inexperienced with examination course, Internal and inexperienced without examination course, Internal and experienced, External:  ")

sensor_2 = input("Please register the type for examiner 2. This can be Internal and inexperienced with examination course, Internal and inexperienced without examination course, Internal and experienced, External:  ")


index_sensor_1_liste = Sensorgrupper_og_navn.index[Sensorgrupper_og_navn['Type sensor'] == sensor_1]
index_sensor_2_liste = Sensorgrupper_og_navn.index[Sensorgrupper_og_navn['Type sensor'] == sensor_2]

print(index_sensor_1_liste)

sensor_1_liste = Sensorgrupper_og_navn.at[index_sensor_1_liste[0], 'Sensorgruppe']
sensor_2_liste = Sensorgrupper_og_navn.at[index_sensor_2_liste[0], 'Sensorgruppe']

print(sensor_1_liste)
print(sensor_2_liste)

clf=DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction_1 = clf.predict([sensor_1_liste])
prediction_2 = clf.predict([sensor_2_liste])

print(prediction_1)
print(prediction_2)

prediction_average = [(g + h) / 2 for g, h in zip(prediction_1, prediction_2)]
print(prediction_average)
prediction_list = np.array(prediction_average).tolist()
pred_a= pd.DataFrame(prediction_list)
pred_a.columns = Labels

print('Prediction: Grade distribution for committee = ' + str(prediction_list))

Y_predict = clf.predict(X)
import numpy as np
y_true = np.array(Y)
y_pred = np.array(Y_predict)
hamming_loss=np.sum(np.not_equal(y_true, y_pred))/float(y_true.size)

print('Hamming loss= '+ str(hamming_loss))

accuracy_score=np.sum(np.equal(y_true, y_pred))/float(y_true.size)

print('Accuracy score= '+ str(accuracy_score))

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

objects = Labels
y_pos = np.arange(len(objects))
pred_average_freq = pred_a.iloc[0]

plt.bar(y_pos, pred_average_freq, align = 'center', alpha=0.5)
plt.xticks(y_pos, Labels, rotation='horizontal')
plt.ylabel('Frequency')
plt.title('Prediction: Grade distribution for committee consisting of an ' + str(sensor_1) + ' examiner ' + 'and an ' + str(sensor_2) + ' examiner')

plt.show()