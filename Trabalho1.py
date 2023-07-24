import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

teste = pd.read_csv('./Data/conjunto_de_teste.csv')
Id = teste["id_solicitante"]
data = pd.read_csv('./Data/conjunto_de_treinamento.csv')

print(data.groupby(["inadimplente"]).mean().T)

line = data.drop(columns = 

              ["inadimplente",
               "id_solicitante",
               "grau_instrucao",
               "estado_onde_nasceu",
               "estado_onde_reside",
               "codigo_area_telefone_residencial",
               "qtde_contas_bancarias_especiais",
               "estado_onde_trabalha",
               "codigo_area_telefone_trabalho",
               "grau_instrucao_companheiro",
               "profissao_companheiro",
               "local_onde_trabalha",
               "tipo_endereco",
               "nacionalidade",
               "possui_email",
               "possui_cartao_visa",
               "possui_cartao_mastercard",
               "possui_cartao_diners",
               "possui_cartao_amex",
               "possui_outros_cartoes",
               "ocupacao"], axis=1)

colum = data["inadimplente"]

line = pd.get_dummies(line, columns = ["forma_envio_solicitacao"])
print(line.T)

for coluna in ["possui_telefone_residencial", 
          "possui_telefone_celular", 
          "vinculo_formal_com_empresa", 
          "possui_telefone_trabalho",
          "sexo"]:

    b = LabelEncoder()

    line[coluna] = b.fit_transform(line[coluna])

print(line.T)

imp = SimpleImputer(strategy='most_frequent')
line = imp.fit_transform(line)

StdSc = StandardScaler()
StdSc = StdSc.fit(line)
line = StdSc.transform(line)

testeLine = teste.drop(columns = 
              ["id_solicitante",
               "grau_instrucao",
               "estado_onde_nasceu",
               "estado_onde_reside",
               "codigo_area_telefone_residencial",
               "qtde_contas_bancarias_especiais",
               "estado_onde_trabalha",
               "codigo_area_telefone_trabalho",
               "grau_instrucao_companheiro",
               "profissao_companheiro",
               "local_onde_trabalha",
               "tipo_endereco",
               "nacionalidade",
               "possui_email",
               "possui_cartao_visa",
               "possui_cartao_mastercard",
               "possui_cartao_diners",
               "possui_cartao_amex",
               "possui_outros_cartoes",
               "ocupacao"], axis=1)

testeLine = pd.get_dummies(testeLine, columns = ["forma_envio_solicitacao"])

for coluna in ["possui_telefone_residencial", 
          "possui_telefone_celular", 
          "vinculo_formal_com_empresa", 
          "possui_telefone_trabalho",
          "sexo"]:
    b = LabelEncoder()
    testeLine[coluna] = b.fit_transform(testeLine[coluna])
    
imp = SimpleImputer(strategy='most_frequent')
testeLine = imp.fit_transform(testeLine)

StdSc = StandardScaler()
StdSc = StdSc.fit(testeLine)
testeLine = StdSc.transform(testeLine)

#Treinamento da IA

trainLine, testLine, trainColum, testColum = train_test_split(line, colum, test_size=0.7)

parametros = {
    "max_depth":[8],
    "n_estimators": [500],
    "min_samples_leaf":[10]
    }

grade = GridSearchCV(RandomForestClassifier(), parametros, cv=10, n_jobs = -1)
grade.fit(trainLine, trainColum)

melhor = grade.best_estimator_
melhor.fit(line,colum)

previsao = melhor.predict(testeLine)

arqPrevisao = pd.DataFrame(previsao, columns=['inadimplente'])
arqPrevisao = pd.concat([Id, arqPrevisao], axis=1)
arqPrevisao = arqPrevisao.to_csv('./Data/ResultadosV3.csv', index=False)
