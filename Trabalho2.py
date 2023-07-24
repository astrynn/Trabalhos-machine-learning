import pandas
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

teste = pandas.read_csv('./Dados/conjunto_de_teste.csv')
data = pandas.read_csv('./Dados/conjunto_de_treinamento.csv')
Id = teste["Id"]

line = data.drop(columns = 
              ['preco',
               'diferenciais',
               'Id',
               'tipo_vendedor',
               'churrasqueira',
               'quadra',
               's_jogos',
               's_ginastica',
               'area_extra',
               'tipo_vendedor'], axis=1)

colum = data['preco']

line = pandas.get_dummies(line, columns = ['tipo'])
line = line.drop(columns = ['tipo_Loft', 'tipo_Quitinete'])
print(line.T)

b = LabelEncoder()
line['bairro'] = b.fit_transform(line['bairro'])

print(line.T)

imp = SimpleImputer(strategy='most_frequent')
line = imp.fit_transform(x)

StdSc = StandardScaler()
StdSc = StdSc.fit(x)
line = StdSc.transform(x)

testeLine = teste.drop(columns = 
              ['diferenciais',
               'Id',
               'tipo_vendedor',
               'churrasqueira',
               'quadra',
               's_jogos',
               's_ginastica',
               'area_extra',
               'tipo_vendedor'], axis=1)

testeLine = pandas.get_dummies(testeLine, columns = ['tipo'])
testeLine = testeLine.drop(columns = ['tipo_Loft'])

b = LabelEncoder()
testeLine['bairro'] = b.fit_transform(testeLine['bairro'])
    
imp = SimpleImputer(strategy='most_frequent')
testeLine = imp.fit_transform(testeLine)

StdSc = StandardScaler()
StdSc = StdSc.fit(testeLine)
testeLine = StdSc.transform(testeLine)

#Treinando a IA

def rmspe(colum, columAnswer):
    result = numpy.sqrt(numpy.mean(numpy.square(((colum - columAnswer) / colum)), axis=0))
    return result

trainLine, testLine, trainColum, testColum = train_test_split(line, colum, test_size=0.2)

regressor = HistGradientBoostingRegressor(l2_regularization=34, max_iter=140, loss = "absolute_error", max_depth=12)
regressor = regressor.fit(trainLine, trainColum)
previsao = regressor.predict(testLine)
erro = mean_squared_error(testColum, previsao)
score = r2_score(testColum, previsao)
print("GB Error was: ", erro)
print("GB Score was: ", score)
print("GB rnmspe score wass: ", rmspe(testColum, previsao))

regressor2 = KNeighborsRegressor(n_neighbors=10, p=1, n_jobs=2, algorithm='kd_tree', weights='distance')
regressor2 = regressor2.fit(trainLine,trainColum)
previsao2 = regressor2.predict(testLine)
erro2 = mean_squared_error(testColum, previsao2)
score2 = r2_score(testColum, previsao2)
print("KNN Error was: ", erro2)
print("KNN Score was: ", score2)
print("KNN rnmspe score was: ", rmspe(testColum, previsao2))

answer = regressor.predict(testeLine)
arqPrevisao = {'Id': Id.index , 'preco': answer}
arqPrevisao = pandas.DataFrame(data=arqPrevisao)

arqPrevisao.to_csv('Resultado.csv', index=False)