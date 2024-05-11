```import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from yellowbrick.model_selection import LearningCurve, ValidationCurve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneOut, cross_val_score
```


```data = pd.read_csv('train.csv')
data
```

**Целью работы** является создание модели машинного обучения для классификации сообщений и спама.    
**Задача** подобрать оптимальную модель машинного обучения для классификации текстов, которая с наибольшей точностью распознает спам-сообщения.      
**Прикладное значение:** модель может быть использована в дальнейшем в различных мессенджерах и электронных почтах

**Описание датасета:**    
Набор данных для классификации сообщений. Датасет состоит из 5574 наблюдений - сообщений. Данные размечены на 2 категории - спам-сообщения('1') и не спам('0'). Этот датасет хорошо подходит для бинарной классификации текстов.

#Выведем общую информацию по данным в датасете:

```data.info()

data.describe()

data.isnull().sum()

data.duplicated().sum()
```

Все ячейки в датасете являеются ненулевыми значениями. Однако в данных присутствуют дубликаты. В дальнейшем их наличие может привести к переобучению модели или влиять на метрики, поэтому целесоообразно их удалить.

```data.drop_duplicates(subset=['sms'], inplace=True)
```
Выведем статистику по количеству спам и не-спам сообщений в датасете

```print(data['label'].value_counts())
```
```spam_count = sum(data.label==True)
non_spam_count = len(data) - spam_count
total_emails = data.shape[0]
```
```categories = ['Спам', 'Не спам']
counts = [spam_count, non_spam_count]
```
```plt.bar(categories, counts, color=['red', 'blue'])
plt.xlabel('Вид сообщения')
plt.ylabel('Количество')
plt.title('Количество спам- и не спам-сообщений')
plt.show()
print("Количество спам-сообщений:", spam_count)
print("Количество не спам-сообщений:", non_spam_count)
print("Всего сообщений:", total_emails)
```
```spam_percent = (spam_count / total_emails) * 100
non_spam_percent = (non_spam_count / total_emails) * 100
labels = 'Спам', 'Не спам'
sizes = [spam_percent, non_spam_percent]
colors = ['red', 'blue']
explode = (0.1, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Процентное соотношение классов')
plt.show()
```
#Предобработка данных

Приведем все данные к одному регистру - нижнему и удалим знаки препинания для дальнейшей работы с датасетом

```data.sms = data.sms.map(lambda t: re.sub(r'[^\w\s]', '', t.lower()))
data.sms = data.sms.map(lambda t: re.sub(r'[^\w\s]', '', t.lower()))

data
```
```nltk.download('punkt')
nltk.download('stopwords')
```
Токенизируем текст сообщений и удалим стоп-слова

```def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

data['clean_txt'] = data['sms'].apply(tokenize_and_remove_stopwords)

data
```
В модели sklearn.feature_extraction есть преобразователь CountVectorizer
с собственными методами лексемизации и нормализации. Метод fit этого преобразователя принимает итерируемую последовательность или список строк
или объектов файлов и создает словарь корпуса. Метод transform преобразует
каждый отдельный документ в разреженный массив, роль индексов в котором
играют кортежи с идентификаторами документов и лексем из словаря, а роль
значений — счетчики лексем.    
Преобразуем текст с помощью этого преобразователя и запишем его в переменную Х


```vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['clean_txt'])

Y = data['label']
```
# Построение моделей классификации

Разобъём выборку на тестовую и обучающую выборки

```X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
```
Построим 7 моделей классификации:    


*   Naive Bayes 
*   DecisionTreeClassifier
*   Support Vector Machine
*   RandomForest
*   KNeighboors
*   RidgeClassifier
*   AdaBoostClassifier









##Найвный Байессовский классификатор

```naive_bayes = MultinomialNB()
start = time.time()
naive_bayes.fit(X_train, y_train)
end = time.time()
time_nb = end-start

time_nb
```
Выведем кривую обучения. На графике мы видим, что как такого переобучения и недообучения нет

```visualizer = LearningCurve(
     MultinomialNB(), scoring='accuracy', train_sizes=np.linspace(0.7, 1.0, 10)
).fit(X_train, y_train).show()

y_pred = naive_bayes.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred)
```
Выведем метрики модели

```accuracy_nb
```
```print(classification_report(y_test, y_pred))
```
Визульное изображение метрик:


ROC-кривая позволяет сравнить различные модели классификации, оценить их производительность и выбрать оптимальный порог для принятия решения о классификации в зависимости от конкретной задачи. Чем ближе кривая к левому верхнему углу, тем лучше производительность модели.    
PR-кривая позволяет оценить производительность классификатора в условиях несбалансированных классов, где точность и полнота играют важную роль. Чем ближе кривая к правому верхнему углу, тем лучше производительность модели. 

classifier = OneVsRestClassifier(MultinomialNB())

# Создаем объект для построения ROC-кривой
```roc_auc_visualizer = ROCAUC(classifier, micro=False, macro=False, per_class=True)

#строим ROC-кривую
roc_auc_visualizer.fit(X_train, y_train)
roc_auc_visualizer.score(X_test, y_test)
roc_auc_visualizer.show()

prc_visualizer = PrecisionRecallCurve(naive_bayes)

prc_visualizer.fit(X_train, y_train)
prc_visualizer.score(X_test, y_test)
prc_visualizer.show()
```
##Дерево решений

```decision_tree = DecisionTreeClassifier()
start = time.time()
decision_tree.fit(X_train, y_train)
end = time.time()
time_dt = end-start
```
Выведем метрики модели

```y_pred = decision_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred)
accuracy_tree

print(classification_report(y_test, y_pred))
```
```visualizer = LearningCurve(
    DecisionTreeClassifier(), scoring='accuracy', train_sizes=np.linspace(0.5, 1.0, 10)
).fit(X_train, y_train).show()
```
Визульное изображение метрик:


ROC-кривая позволяет сравнить различные модели классификации, оценить их производительность и выбрать оптимальный порог для принятия решения о классификации в зависимости от конкретной задачи. Чем ближе кривая к левому верхнему углу, тем лучше производительность модели.    
PR-кривая позволяет оценить производительность классификатора в условиях несбалансированных классов, где точность и полнота играют важную роль. Чем ближе кривая к правому верхнему углу, тем лучше производительность модели.

```classifier = OneVsRestClassifier(DecisionTreeClassifier())

# Создаем объект для построения ROC-кривой
roc_auc_visualizer = ROCAUC(classifier, micro=False, macro=False, per_class=True)

#строим ROC-кривую
roc_auc_visualizer.fit(X_train, y_train)
roc_auc_visualizer.score(X_test, y_test)
roc_auc_visualizer.show()

prc_visualizer = PrecisionRecallCurve(decision_tree)

prc_visualizer.fit(X_train, y_train)
prc_visualizer.score(X_test, y_test)
prc_visualizer.show()
```
##Метод опорных векторов

```svc = SVC()
start = time.time()
svc.fit(X_train, y_train)
end = time.time()
time_svc = end-start
y_pred = svc.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred)
accuracy_svc
```
```visualizer = LearningCurve(
    SVC(), scoring='accuracy', train_sizes=np.linspace(0.7, 1.0, 10)
).fit(X_train, y_train).show()
```
```classifier = OneVsRestClassifier(SVC())

# Создаем объект для построения ROC-кривой
roc_auc_visualizer = ROCAUC(classifier, micro=False, macro=False, per_class=True)

#строим ROC-кривую
roc_auc_visualizer.fit(X_train, y_train)
roc_auc_visualizer.score(X_test, y_test)
roc_auc_visualizer.show()

prc_visualizer = PrecisionRecallCurve(svc)

prc_visualizer.fit(X_train, y_train)
prc_visualizer.score(X_test, y_test)
prc_visualizer.show()
```
##Случайный лес

```random_forest = RandomForestClassifier(random_state=42)
start = time.time()
random_forest.fit(X_train,y_train)
end = time.time()
time_rf = end-start
y_pred = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred)
accuracy_rf
```
```print(classification_report(y_test, y_pred))
```
```visualizer = LearningCurve(
    RandomForestClassifier(), scoring='accuracy', train_sizes=np.linspace(0.7, 1.0, 10)
).fit(X_train, y_train).show()
```
```classifier = OneVsRestClassifier(RandomForestClassifier())

# Создаем объект для построения ROC-кривой
roc_auc_visualizer = ROCAUC(classifier, micro=False, macro=False, per_class=True)

#строим ROC-кривую
roc_auc_visualizer.fit(X_train, y_train)
roc_auc_visualizer.score(X_test, y_test)
roc_auc_visualizer.show()

prc_visualizer = PrecisionRecallCurve(random_forest)

prc_visualizer.fit(X_train, y_train)
prc_visualizer.score(X_test, y_test)
prc_visualizer.show()
```
## Метод k-ближайших соседей

```k_neighbors = KNeighborsClassifier()
start = time.time()
k_neighbors.fit(X_train, y_train)
end = time.time()
time_kn = end-start

y_pred = k_neighbors.predict(X_test)
accuracy_kn = accuracy_score(y_test, y_pred)
accuracy_kn
```
```print(classification_report(y_test, y_pred))
```
```visualizer = LearningCurve(
   KNeighborsClassifier(), scoring='accuracy', train_sizes=np.linspace(0.7, 1.0, 10)
).fit(X_train, y_train).show()
```
```classifier = OneVsRestClassifier(KNeighborsClassifier())

# Создаем объект для построения ROC-кривой
roc_auc_visualizer = ROCAUC(classifier, micro=False, macro=False, per_class=True)

#строим ROC-кривую
roc_auc_visualizer.fit(X_train, y_train)
roc_auc_visualizer.score(X_test, y_test)
roc_auc_visualizer.show()

prc_visualizer = PrecisionRecallCurve(k_neighbors)

prc_visualizer.fit(X_train, y_train)
prc_visualizer.score(X_test, y_test)
prc_visualizer.show()
```
## Гребневой классификатор

```ridge = RidgeClassifier()
start = time.time()
ridge.fit(X_train, y_train)
end = time.time()
time_rc = end-start
```
```y_pred = ridge.predict(X_test)
accuracy_rc = accuracy_score(y_test, y_pred)
accuracy_rc
```
```print(classification_report(y_test, y_pred))
```
```visualizer = LearningCurve(
   RidgeClassifier(), scoring='accuracy', train_sizes=np.linspace(0.7, 1.0, 10)
).fit(X_train, y_train).show()
```
```classifier = OneVsRestClassifier(RidgeClassifier())

# Создаем объект для построения ROC-кривой
roc_auc_visualizer = ROCAUC(classifier, micro=False, macro=False, per_class=True)

#строим ROC-кривую
roc_auc_visualizer.fit(X_train, y_train)
roc_auc_visualizer.score(X_test, y_test)
roc_auc_visualizer.show()

prc_visualizer = PrecisionRecallCurve(ridge)

prc_visualizer.fit(X_train, y_train)
prc_visualizer.score(X_test, y_test)
prc_visualizer.show()
```
## AdaBoostClassifier

```adaboost = AdaBoostClassifier()
start = time.time()
adaboost.fit(X_train, y_train)
end = time.time()
time_adb = end-start
```
```y_pred = adaboost.predict(X_test)
accuracy_adb = accuracy_score(y_test, y_pred)
accuracy_adb
```
```print(classification_report(y_test, y_pred))
```
```visualizer = LearningCurve(
   RidgeClassifier(), scoring='accuracy', train_sizes=np.linspace(0.7, 1.0, 10)
).fit(X_train, y_train).show()
```
```classifier = OneVsRestClassifier(RidgeClassifier())

# Создаем объект для построения ROC-кривой
roc_auc_visualizer = ROCAUC(classifier, micro=False, macro=False, per_class=True)

#строим ROC-кривую
roc_auc_visualizer.fit(X_train, y_train)
roc_auc_visualizer.score(X_test, y_test)
roc_auc_visualizer.show()

prc_visualizer = PrecisionRecallCurve(ridge)

prc_visualizer.fit(X_train, y_train)
prc_visualizer.score(X_test, y_test)
prc_visualizer.show()
```
#Сравнение моделей

```def measure(name, y_test, y_test_pred):
  metrics = pd.DataFrame({
    name: [
        accuracy_score(y_test, y_test_pred),
        precision_score(y_test, y_test_pred),
        recall_score(y_test, y_test_pred),
        f1_score(y_test, y_test_pred),
    ],
  }, index = ["Accuracy", "Precision", "Recall", "F1"])
  return metrics
```
```models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Ridge Classifier": RidgeClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(measure(name, y_test, y_pred))
```
Поскольку мы имеем дело с несбалансированными классами, в нашем случае важно больше сравнивать метрики Precision и Recall, при этом учитывая Accuracy 

```time = pd.DataFrame({'модель':['DecisionTree', 'K-neghbors', 'Naive Bayes', 'Ridge Classifier',
                               'RandomForest', 'SVC', 'AdaBoost'],
                      'time':[time_dt,time_kn,time_nb,
                              time_rc,time_rf, time_svc, time_adb]})

time
```
Выбирая наилучшую модель, мы сравниваем две модели с более хорошими метриками - Naive Bayes, Support Vector Machine, Ranom Forest и Ridge Classifier. У Байесовского классификатора лучше метрика Recall, что показывает, что модель более полно обнаруживает спам. Это важно для минимизации пропуска спама и обеспечения высокой степени обнаружения нежелательных сообщений. У гребневого классификатора, метода опорных векторов и случайного леса более высокая метрика Precision, что показывает, что модель более точно определяет спам. Это важно для предотвращения ложных срабатываний и ошибочного определения нормальных сообщений как спама. Для нашей задачи более значимо избежать ложных блокировок важных сообщений, исходя из этого наилучшая модель, у которой более высокая accuracy и precision и неплохие другие метрики, - RidgeClassifier. 

#Анализ устойчивости модели. Изменение модели для её усовершенствования

Для оценки устойчивости модели к изменениям в данных, мы можем использовать метод кросс-валидации:

Применим Stratified валидацию, т.к. у нас несбалансированные классы

```skf = StratifiedKFold(n_splits=3,shuffle=True, random_state=15)
skf.get_n_splits(X, Y)
```
```skf.split(X,Y)
```
```cv_results = cross_val_score(ridge,                  # модель
                             X,                      # матрица признаков
                             Y,                      # вектор цели
                             cv = skf,           # тип разбиения
                             scoring = 'accuracy',   # метрика
                             n_jobs=-1)              # используются все ядра CPU

print("Кросс-валидация: ", cv_results)
print("Среднее по кросс-валидации: ", cv_results.mean())
```
Модель может быть применена в реальных условиях, так как точность модели на тестовом наборе данных
хорошая и близка к точности, полученной на кросс-валидации

RidgeClassifier - Классификатор, использующий гребневую регрессию.

Этот классификатор сначала преобразует целевые значения в {-1, 1}, а затем обрабатывает проблему как задачу регрессии.

Параметры модели:
alpha, fit_intercept,copy_X, max_iter, tol, class_weight, solver, positive, random_state

Выведем наилучшую модель 

```param_grid = {'alpha':[0.1,1,5,10,50,100], 'fit_intercept':[True, False], 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']}

grid_model = GridSearchCV(estimator=ridge,
                          param_grid=param_grid,
                          scoring='accuracy',
                          cv=5,
                          verbose=2)

grid_model.fit(X_train,y_train)
```
```grid_model.best_estimator_
```
Оптимальные параметры:

```grid_model.best_params_
```
```ridge_new = RidgeClassifier(alpha=5,fit_intercept=True, solver='auto')
ridge_new.fit(X_train, y_train)

y_pred = ridge_new.predict(X_test)
accuracy_rc_n = accuracy_score(y_test, y_pred)
pr_n = precision_score(y_test, y_pred),
rc_n = recall_score(y_test, y_pred),
f1_n = f1_score(y_test, y_pred)
accuracy_rc_n, pr_n, rc_n, f1_n
```
```print(classification_report(y_test, y_pred))
```
В данной задаче классификации спама была проведена успешная работа, направленная на
разделение сообщений на два класса: спам и не спам. Результаты этой задачи имеют
важное значение для различных приложений-мессенджеров. 
В ходе исследования были использованы следующие алгоритмы машинного обучения: Naive Bayes, DecisionTreeClassifier,Support Vector Machine,RandomForest,KNeighboors,RidgeClassifier, AdaBoostClassifier.     

Для каждого
алгоритма было оценено качество модели с помощью метрик accuracy, precision, recall, F1-score.
Было выявлено, что алгоритм RidgeClassifier показывает наилучшее качество, а K-Nearest Neighbors - наихудшее качество для данной задачи классификации спама.
Таким образом, результаты исследования показали, что машинное обучение может быть эффективно
использовано для классификации текстовых данных, в частности, для классификации
спам-сообщений. При этом выбор алгоритма машинного обучения имеет большое значение для достижения
высокого качества предсказания.

