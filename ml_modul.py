from copy import copy

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder


class MlHandler(object):

    def __init__(self, source):
        self.source = source

    def drop_na(self, columns=None, axis=0, inplace=False):
        if not columns:
            columns = self.source.columns.tolist()
        elif type(columns) == str:
            columns = [columns]

        new_dataframe = copy(self.source)

        for index, value in new_dataframe.iterrows():
            for column in columns:
                if pd.isna(value[column]):
                    if axis == 0:
                        new_dataframe.drop([index], inplace=True)
                        break
                    elif axis == 1:
                        del new_dataframe[column]
                        columns.remove(column)

        if inplace:
            self.source = new_dataframe
        else:
            return new_dataframe

    def replace(self, old, new, columns=None, inplace=False):
        if not columns:
            columns = self.source.columns.tolist()
        elif type(columns) == str:
            columns = [columns]

        new_dataframe = copy(self.source)

        for index, value in new_dataframe.iterrows():
            for column in columns:
                if pd.isna(old) and pd.isna(value[column]):
                    new_dataframe[column].iloc[index] = new
                elif value[column] == old:
                    new_dataframe[column].iloc[index] = new
        if inplace:
            self.source = new_dataframe
        else:
            return new_dataframe
        
    def replace_textdata(self, pre_column, column, num_column, word = ''):
        
        df = self.source
        
        if word == '':
            df_miss_EFT = df[pd.isna(df[column])]
        else:
            df_miss_EFT = df[df[column] == word]
        model_miss_value = df_miss_EFT[pre_column].unique()
        model_is_value = {}
        for row in model_miss_value:
            if word == '':
                model_is_value.update({row:df[(pd.notnull(df[column])) & (df[pre_column]==row)].iloc[0, num_column]})
            else: model_is_value.update({row:df[(df[column] != word) & (df[pre_column]==row)].iloc[0, num_column]})
        miss_EFT = {}
        for model in  model_miss_value:
            indexes = []
            for index, _ in df_miss_EFT[df_miss_EFT[pre_column] == model].iterrows():
                indexes.append(index)
            miss_EFT.update({model: indexes})
        for key, value in miss_EFT.items():
            for i in value:
                df[column].iloc[i] = model_is_value.get(key)
        return df
    
    def replace_by_mode(self, old, columns=None, inplace=False):

        if not inplace:
            new_df = copy(self.source)

        if not columns:
            columns = self.source.columns.tolist()
        elif type(columns) == str:
            columns = [columns]

        mode = self.source.mode().iloc[0]

        for column in columns:
            result = self.replace(old, mode[column], columns=column, inplace=inplace)

            if not inplace and result is not None:
                new_df[column] = result[column]

        if not inplace:
            return new_df

    def replace_by_median(self, old, columns=None, inplace=False):

        if not inplace:
            new_df = copy(self.source)

        median = self.source.median()

        if not columns:
            columns = median.index.tolist()
        elif type(columns) == str:
            columns = [columns]

        for column in columns:
            try:
                result = self.replace(old, median[column], columns=column, inplace=inplace)

                if not inplace and result is not None:
                    new_df[column] = result[column]
            except:
                print("Columns have to be numerical")
                return

        if not inplace:
            return new_df

    def replace_by_avg(self, old, columns=None, inplace=False):
        if not inplace:
            new_df = copy(self.source)

        mean = self.source.mean()
        if not columns:
            columns = mean.index.tolist()
        elif type(columns) == str:
            columns = [columns]

        for column in columns:
            try:
                result = self.replace(old, mean[column], columns=column, inplace=inplace)

                if not inplace and result is not None:
                    new_df[column] = result[column]
            except:
                print("Columns have to be numerical")
                return

        if not inplace:
            return new_df

    def linear_replace(self, inplace=False, columns=None):
        if not inplace:
            new_df = copy(self.source)

        numeric_data = self.source.select_dtypes(include=['float', 'int'])
        linear_regression = LinearRegression()

        if not columns:
            columns = numeric_data.select_dtypes(include=['float', 'int']).columns.tolist()
        elif type(columns) == str:
            columns = [columns]

        for column in columns:

            train_df = numeric_data.dropna(axis=0)
            y_train = train_df[column]
            train_df.drop(column, axis=1, inplace=True)
            try:
                linear_regression.fit(train_df, y_train)

                y_pred = linear_regression.predict(numeric_data[numeric_data[column].isnull()].drop(column, axis=1))
            except:
                print("Columns have to be numerical")
                return

            i = 0

            for index, value in self.source.iterrows():

                if pd.isna(value[column]):
                    if not inplace:
                        new_df[column].iloc[index] = y_pred[i]
                    else:
                        self.source[column].iloc[index] = y_pred[i]

                    i += 1

        if not inplace:
            return new_df

    def knn_replace(self, inplace=False, columns=None):

        if not inplace:
            new_df = copy(self.source)

        numeric_data = self.source.select_dtypes(include=['float', 'int'])

        neigh = KNeighborsClassifier(p=1)

        if not columns:
            columns = numeric_data.select_dtypes(include=['float', 'int']).columns.tolist()
        elif type(columns) == str:
            columns = [columns]

        for column in columns:

            train_df = numeric_data.dropna(axis=0)
            y_train = train_df[column]
            train_df.drop(column, axis=1, inplace=True)

            neigh.fit(train_df, y_train)

            y_pred = neigh.predict(numeric_data[numeric_data[column].isnull()].drop(column, axis=1))


            i = 0

            for index, value in self.source.iterrows():

                if pd.isna(value[column]):
                    if not inplace:
                        new_df[column].iloc[index] = y_pred[i]
                    else:
                        self.source[column].iloc[index] = y_pred[i]

                    i += 1

        if not inplace:
            return new_df

    def normalize(self, inplace=False):
        if not inplace:
            new_df = copy(self.source)

        numeric_data = self.source.select_dtypes(include=['float', 'int'])

        columns = numeric_data.columns.tolist()

        for column in columns:
            mean = numeric_data[column].mean()
            std = np.std(numeric_data[column])
            numeric_data[column] = numeric_data[column] - mean
            numeric_data[column] /= std
            if not inplace:
                new_df[column] = numeric_data[column]

        if inplace:
            self.source = new_df
        else:
            return new_df

    def scale(self, inplace=False):
        if not inplace:
            new_df = copy(self.source)
        numeric_data = self.source.select_dtypes(include=['float', 'int'])

        columns = numeric_data.columns.tolist()

        for column in columns:
            numeric_data[column] = (numeric_data[column] - numeric_data[column].min()) / (
                    numeric_data[column].max() - numeric_data[column].min())
            if not inplace:
                new_df[column] = numeric_data[column]

        if inplace:
            self.source = new_df
        else:
            return new_df

    def distance(self, columns, pred_data=None, source=None):
        dataframe = copy(self.source[columns]) if source is not None else source
        distance_list = []

        for ind, val in dataframe.iterrows():
            distance = 0
            for column in columns:
                if pd.notna(pred_data[column]):
                    distance += np.square(pred_data[column] - val[column])
            distance_list.append(np.sqrt(distance))
        dataframe['Distance'] = distance_list
        return dataframe.sort_values(["Distance"], kind='heapsort')

    def knn_replace(self, inplace=False, columns=None):

        if not inplace:
            new_df = copy(self.source)

        numeric_data = self.source.select_dtypes(include=['float64', 'int64'])

        if not columns:
            columns = numeric_data.columns.tolist()
        elif type(columns) == str:
            columns = [columns]

        for column in columns:
            df_to_predict = numeric_data[numeric_data[column].isnull()]

            train_df = numeric_data[column].notna()
            for index, value in df_to_predict.iterrows():
                shortest_df = self.distance(columns, value, train_df)[1:self.__k + 1]
                df_to_predict.Litres.loc[index] = shortest_df[column].mean()

            i = 0

            for index, value in self.source.iterrows():
                if pd.isna(value[column]):
                    if not inplace:
                        new_df[column].iloc[index] = df_to_predict.Litres.iloc[i]
                    else:
                        self.source[column].iloc[index] = df_to_predict.Litres.iloc[i]

                    i += 1

        if not inplace:
            return new_df

# ml_handler = MlHandler(pd.read_csv('test_dataset.csv'))
# print(ml_handler.source)
# print(ml_handler.drop_na(columns=['ThirdRow', 'FirstRow'], axis=1, inplace=False))
# print(ml_handler.replace(None, 'not', inplace=True))
# print(ml_handler.replace_by_mode(None, columns=['ThirdRow', 'FirstRow'], inplace=True))
# print(ml_handler.replace_by_median(None))
# print(ml_handler.replace_by_avg(None, columns='FirstRow'))
# print(ml_handler.linear_replace())
# print(ml_handler.knn_replace())