from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


class SimpleImputer_(BaseEstimator, TransformerMixin):
    def __init__(self, ls_features, strategy='mean'):
        self.ls_features = ls_features
        self.strategy = strategy

    def fit(self, X, y):
        df = X.copy()
        self.imputer = SimpleImputer(missing_values=np.nan, strategy=self.strategy)
        self.imputer.fit(df[self.ls_features])

        return self

    def transform(self, X):
        # X_ = X.copy()
        X[self.ls_features] = self.imputer.transform(X[self.ls_features])

        return X

class OneHotEncoderFeatures_(BaseEstimator, TransformerMixin):
    """
    Classe genérica para criar novas features dummies a partir de features selecionadas.

    Parâmetros:
        ls_features: Lista de features que serão transformadas em dummies.
        nulls_to_base: Boleano utilizado para casos onde queremos categorias nulas para base das dummies.
        handle_unknown: Especifica a maneira para lidar com categorias não definadas no fit ('error','ignore','infrequent_if_exist').

    Exemplo de uso:
        one = OneHotEncoderFeatures(ls_features=['feature_1','feature_2'], nulls_to_base=True)
        one.fit(X, y)
        one.transform(X)
    """
    def __init__(self, ls_features, nulls_to_base=False, handle_unknown='error', target='target'):
        self.ls_features = ls_features
        self.nulls_to_base = nulls_to_base
        self.handle_unknown = handle_unknown
        self.target = target
        
    # Functions to get features with nulls and types
    def get_features_with_null(self, df, ls_features):
        # De/para de tipo de dado e o tipo de nulo
        de_para_nulls ={'object':None, 'float64':np.nan, 'Int64':np.nan, 'bool':pd.NA}
        # Dict with count of nulls by column
        dict_count_nulls = df[ls_features].isnull().sum().to_dict()
        # List with features with null
        features_nulls = [i for i in dict_count_nulls if dict_count_nulls[i] > 0]
        # List of types 
        lst_types = [str(i) for i in df[features_nulls].dtypes]
        # List of nulls
        lst_nulls = [de_para_nulls[i] for i in lst_types]
        # Dict with features nulls and value of null
        dict_results = {'features_nulls':features_nulls, 'type_nulls':lst_nulls}

        return dict_results
    
    # função para concatenar as features com aquelas que não passaram pelo processo de encoder
    def concat_with_rest(self, df, df_transformed, ls_features):              
        # get the rest of the features
        outras_features = [feature for feature in df.columns if feature not in ls_features]
        # concaternar o restante das features com as features que passaram pelo one-hot-encoding
        df_concat = pd.concat([df_transformed, df[outras_features]],axis=1)
        return df_concat
    
    # função para coletar a categoria de cada feature com a proporção de sucesso para práxima da proporção de sucesso do target
    def get_category_by_ratio_positive(self, df, ls_features, target):
        "Target coluna binária"
        # Dicionário onde será salvo a categoria de cada feature 
        dict_fetures_categories = {}
        
        for f in ls_features:
            value_positive = float(df[target].value_counts(normalize=True).to_frame().reset_index().query("index == 1")[target]) # proporção de sucesso
            df_cat = df.groupby(f, dropna=False)[target].sum().to_frame() / df[target].sum() # proporção de sucesso por categoria da feature
            df_cat['dist'] = np.sqrt((df_cat[target] - value_positive)**2) # distância
            df_cat = df_cat.sort_values('dist') # ordenando da menor distância para a maior
            category = df_cat.iloc[0].name # categoria da feature com a proporção mais próxima da proporção de sucesso do target
            # Salvando o resultado no dicionário
            dict_fetures_categories[f] = category
        
        return {'features':list(dict_fetures_categories.keys()), 'categories':list(dict_fetures_categories.values())}
        
    
    def fit(self, X, y):
        df = X.copy()
        df[self.target] = y
        
        if self.nulls_to_base:
            # Dicionário com features com nulos
            dict_categories_nulls = self.get_features_with_null(df, self.ls_features)
            # Lista de features sem nulos
            lst_others_features = [f for f in self.ls_features if f not in dict_categories_nulls['features_nulls']]
            # Dicionário com features sem nulos
            dict_categories_ratio_sucess = self.get_category_by_ratio_positive(df, lst_others_features, self.target)
            
            # Lista com features com nulas e não nulas
            self.all_features = dict_categories_nulls['features_nulls'] + dict_categories_ratio_sucess['features']
            # Lista de categorias de base de cada feature
            self.all_categories = dict_categories_nulls['type_nulls'] + dict_categories_ratio_sucess['categories']
            
            self.onehot = OneHotEncoder(handle_unknown=self.handle_unknown, drop=self.all_categories)
            self.onehot.fit(df[self.all_features])
        else:
             # Dicionário com features e categorias com taxa de positivo mais próxima da proporção de positivos do target
            dict_categories_ratio_sucess = self.get_category_by_ratio_positive(df, self.ls_features, self.target)
            self.all_features = dict_categories_ratio_sucess['features']
            self.all_categories = dict_categories_ratio_sucess['categories']
            
            self.onehot = OneHotEncoder(handle_unknown=self.handle_unknown, drop=self.all_categories)
            self.onehot.fit(df[self.all_features])
        
        return self
    
    def transform(self, X):
        
        # obtendo as categorias
        feature_names = ['OHE_' + str(col)  for col in self.onehot.get_feature_names_out(self.all_features)]
        # Transformando o dataframe
        X_transformed = pd.DataFrame(
            self.onehot.transform(X[self.all_features]).toarray(),
            columns= feature_names,
            index=X.index
        )
        # Unindo o dataframe transformado com o restante das features não transformadas
        X_ = self.concat_with_rest(df=X, df_transformed=X_transformed, ls_features=self.all_features)

        return X_

class GenericNumericEstimator_(BaseEstimator, TransformerMixin):
    """
    Classe genérica para criar novas features dummies a partir de features selecionadas.

    Parâmetros:
        ls_features: Lista de features que serão transformadas em dummies.
        nulls_to_base: Boleano utilizado para casos onde queremos categorias nulas para base das dummies.
        handle_unknown: Especifica a maneira para lidar com categorias não definadas no fit ('error','ignore','infrequent_if_exist').

    Exemplo de uso:
        one = OneHotEncoderFeatures(ls_features=['feature_1','feature_2'], nulls_to_base=True)
        one.fit(X, y)
        one.transform(X)
    """
    def __init__(self, estimator, type_numeric=['floating'], features_to_ignore=[]):
        self.estimator = estimator
        self.type_numeric = type_numeric
        self.features_to_ignore = features_to_ignore
        self.modified_features = []
    
    def fit(self, X, y=None):
        if self.estimator is not None:
            self.column_types_ = {str(col):str(pd.api.types.infer_dtype(X[col])) for col in X.columns}
            self.num_cols_ = [k for k in self.column_types_ if self.column_types_[k] in self.type_numeric]
            self.num_cols_ = [col for col in self.num_cols_ if col not in self.features_to_ignore]
            self.modified_features.extend(self.num_cols_)
            self.estimator.fit(X[self.num_cols_], y)
        return self
    
    # função para concatenar as features com aquelas que não passaram pelo processo de encoder
    def concat_with_rest(self, df, df_transformed):              
        # get the rest of the features
        outras_features = [feature for feature in df.columns if feature not in self.num_cols_]
        # concaternar o restante das features com as features que passaram pelo one-hot-encoding
        df_concat = pd.concat([df_transformed, df[outras_features]],axis=1)
        return df_concat
    
    def transform(self, X):
        if self.estimator is not None:
            X_transformed = pd.DataFrame(self.estimator.transform(X[self.num_cols_]), index=X.index, columns = self.num_cols_)
            X_ = self.concat_with_rest(df=X, df_transformed=X_transformed)[X.columns]
        return X_
