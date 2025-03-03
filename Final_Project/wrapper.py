import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer
from sklearn.impute import KNNImputer
import shap
import category_encoders as ce
from scipy.stats import skew
from statsmodels.api import OLS, add_constant

class CustomDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return CustomDataFrame

    def tips(self):
        print("\n\033[1mAdvantages and disadvantages of the different methods to consider:\033[0m")
        print("""\033[1mRandom:\033[0m This was the baseline method. It is simple and fast, but it does not take into account the relationships between the attributes and the distribution of the data.\nIf you have a large dataset and the missing values are not significant, this method can be used.\nAlso, if the results of the other methods are not significantly better, this method can be used as a fallback.
              
\033[1mMean:\033[0m This method is simple and fast. It is useful when the data is normally distributed and the missing values are not significant. However, it can cause some distortion in the data. 
Let's consider an example of cars dataset and say the horsepower attribute has missing values. If we fill the missing values with the mean of the horsepower, it may distort the data because both a luxury car and an economy car will get the same horsepower value.
                
\033[1mMedian:\033[0m This method is similar to the mean method, but it is more robust to outliers. It is useful when the data is skewed and the missing values are not significant. However, it can also cause some distortion in the data. 
                
\033[1mFrequent:\033[0m This method is useful when the data is categorical and the missing values are not significant. It is simple and fast, but it can cause some distortion in the data.
                
\033[1mKNN:\033[0m This method is useful when the data has a complex relationship between the attributes and the missing values are significant. It takes into account the relationships between the attributes and the distribution of the data. However, it is computationally expensive and may not work well with high-dimensional data.
                
\033[1mLinear Regression:\033[0m This method is useful when the data has a linear relationship between the attributes and the missing values are significant. It takes into account the relationships between the attributes and the distribution of the data. However, it may not work well with non-linear data and may cause overfitting of the data.
                
\033[1mDrop:\033[0m This method is useful when the missing values are insignificant and cannot be imputed. It is simple and fast, but it can cause a loss of information. It should be used when the amount of missing values is small or as a last resort.""")

    def show_nulls(self, df_test=None):
        print('Train data:')
        train_nulls = self.isnull().sum()
        train_nulls_percentage = (train_nulls / len(self)) * 100
        train_nulls_percentage = train_nulls_percentage.round(3)
        train_nulls_percentage = train_nulls_percentage.astype(str) + '%'
        train_nulls_info = pd.concat([train_nulls[train_nulls > 0], train_nulls_percentage[train_nulls > 0]], axis=1)
        train_nulls_info.columns = ['Null Count', 'Percentage']
        print(train_nulls_info)
        
        if df_test is not None:
            print()
            print('Test data:')
            test_nulls = df_test.isnull().sum()
            test_nulls_percentage = (test_nulls / len(df_test)) * 100
            test_nulls_percentage = test_nulls_percentage.round(3)
            test_nulls_percentage = test_nulls_percentage.astype(str) + '%'
            test_nulls_info = pd.concat([test_nulls[test_nulls > 0], test_nulls_percentage[test_nulls > 0]], axis=1)
            test_nulls_info.columns = ['Null Count', 'Percentage']
            print(test_nulls_info)
    
    def create_mcar(self, df_test, attributes, percentages):
        for attribute, percentage in zip(attributes, percentages):
            percentage = float(percentage) / 100  # Convert percentage to a fraction
            
            # Randomly select rows to set as NaN
            missing_train = np.random.choice(self.index, 
                                             size=int(percentage * len(self)), 
                                             replace=False)
            missing_test = np.random.choice(df_test.index, 
                                            size=int(percentage * len(df_test)), 
                                            replace=False)
            
            self.loc[missing_train, attribute] = np.nan
            df_test.loc[missing_test, attribute] = np.nan

        return self, df_test, missing_train, missing_test

    def drop_missing_values(self, df_test, attribute):
        self.dropna(subset=[attribute], inplace=True)
        df_test.dropna(subset=[attribute], inplace=True)
    
    def fill_randomly(self, df_test, attribute):
        min_val = self[attribute].min()
        max_val = self[attribute].max()
        
        self[attribute] = self[attribute].fillna(np.random.randint(min_val, max_val))
        df_test[attribute] = df_test[attribute].fillna(np.random.randint(min_val, max_val))
    
    def fill_with_mean(self, df_test, attribute):
        mean_val = self[attribute].mean()
        
        self[attribute] = self[attribute].fillna(mean_val)
        df_test[attribute] = df_test[attribute].fillna(mean_val)
    
    def fill_with_median(self, df_test, attribute):
        median_val = self[attribute].median()
        
        self[attribute] = self[attribute].fillna(median_val)
        df_test[attribute] = df_test[attribute].fillna(median_val)
    
    def fill_with_freq(self, df_test, attribute):
        freq_val = self[attribute].mode()[0] # mode returns a series, so we need to get the first value
        
        self[attribute] = self[attribute].fillna(freq_val)
        df_test[attribute] = df_test[attribute].fillna(freq_val)
    
    def fill_with_knn(self, df_test, attribute, is_categorical):
        if is_categorical:
            # Convert categorical data to numerical data using OrdinalEncoder
            encoder = OrdinalEncoder()
            self[attribute] = encoder.fit_transform(self[[attribute]])
            df_test[attribute] = encoder.transform(df_test[[attribute]])
        
        imputer = KNNImputer(n_neighbors=3, weights='uniform', metric='nan_euclidean')
        
        self[attribute] = imputer.fit_transform(self[[attribute]])
        df_test[attribute] = imputer.transform(df_test[[attribute]])
        
        if is_categorical:
            # Convert numerical data back to categorical data
            self[attribute] = encoder.inverse_transform(self[[attribute]])
            df_test[attribute] = encoder.inverse_transform(df_test[[attribute]])
    
    def fill_with_linear_regression(self, df_test, attribute, target):
        self_dummies = pd.get_dummies(self, drop_first=True)
        df_test_dummies = pd.get_dummies(df_test, drop_first=True)
        
        # Split the data into two sets: one with missing values and one without
        df_train_missing = self_dummies[self_dummies[attribute].isnull()]
        df_train_not_missing = self_dummies.dropna(subset=[attribute])
        
        # Split the data into X and y
        X_train = df_train_not_missing.drop(columns=[attribute, target])
        # Fill missing values with '-1'
        X_train = X_train.fillna(-1)
        y_train = df_train_not_missing[attribute]
        
        # Train a linear regression model
        linear_reg = LinearRegression()
        linear_reg.fit(X_train, y_train)
        
        # Predict the missing values
        X_test = df_train_missing.drop(columns=[attribute, target])
        X_test = X_test.fillna(-1)
        y_pred = linear_reg.predict(X_test)
        
        # Fill the missing values
        self.loc[self[attribute].isnull(), attribute] = y_pred
        
        # Repeat the process for the test data
        df_test_missing = df_test_dummies[df_test_dummies[attribute].isnull()]
        df_test_not_missing = df_test_dummies.dropna(subset=[attribute])
        
        X_test = df_test_not_missing.drop(columns=[attribute, target])
        # Fill missing values with '-1'
        X_test = X_test.fillna(-1)
        y_test = df_test_not_missing[attribute]
        y_test = y_test.fillna(-1)
        
        linear_reg.fit(X_test, y_test)
        
        X_test = df_test_missing.drop(columns=[attribute, target])
        X_test = X_test.fillna(-1)
        y_pred = linear_reg.predict(X_test)
        
        df_test.loc[df_test[attribute].isnull(), attribute] = y_pred

    def train_and_predict(self, df_test, target):
        X_train = self.drop(columns=target)
        encoder = ce.OrdinalEncoder()
        # If there are any categorical columns, convert them to numerical using one-hot encoding
        if self.select_dtypes(include='object').shape[1] > 0:
            X_train = encoder.fit_transform(X_train)
        X_train = X_train.fillna(-1)
        
        y_train = self[target]
        X_test = df_test.drop(columns=target)
        # If there are any categorical columns, convert them to numerical using one-hot encoding
        if df_test.select_dtypes(include='object').shape[1] > 0:
            X_test = encoder.transform(X_test)
        X_test = X_test.fillna(-1)
        y_test = df_test[target]
        
        linear_reg = LinearRegression()
        linear_reg.fit(X_train, y_train)
        y_pred = linear_reg.predict(X_test)
        
        return y_test, y_pred

    def eval_r2(self, y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        return r2

    def eval_mse(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        return mse

    def eval_mape(self, y_test, y_pred):
        mape = mean_absolute_percentage_error(y_test, y_pred)
        return mape

    def eval_mae(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        return mae

    def eval_rmse(self, y_test, y_pred):
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse

    def try_all_methods(self, df_test, attribute, target, is_categorical):
        df_train_lr = self.copy()
        df_test_lr = df_test.copy()
        df_train_lr.fill_with_linear_regression(df_test_lr, attribute, target)
        
        df_train_r = self.copy()
        df_test_r = df_test.copy()
        df_train_r.fill_randomly(df_test_r, attribute)
        
        df_train_mean = self.copy()
        df_test_mean = df_test.copy()
        df_train_mean.fill_with_mean(df_test_mean, attribute)
        
        df_train_median = self.copy()
        df_test_median = df_test.copy()
        df_train_median.fill_with_median(df_test_median, attribute)

        df_train_freq = self.copy()
        df_test_freq = df_test.copy()
        df_train_freq.fill_with_freq(df_test_freq, attribute)
        
        df_train_knn = self.copy()
        df_test_knn = df_test.copy()
        df_train_knn.fill_with_knn(df_test_knn, attribute, is_categorical)
        
        df_train_drop = self.copy()
        df_test_drop = df_test.copy()
        df_train_drop.drop_missing_values(df_test_drop, attribute)
        
        df_array = [df_train_r, df_test_r, df_train_mean, df_test_mean, df_train_median, df_test_median, df_train_freq, df_test_freq, df_train_knn, df_test_knn, df_train_lr, df_test_lr, df_train_drop, df_test_drop]
        return df_array

    def compare_fills(self, df_array, target):
        methods = ['Random', 'Mean', 'Median', 'Frequent', 'KNN', 'Linear Regression', 'Drop']
        r2_scores = np.zeros(len(df_array) // 2)
        mse_scores = np.zeros(len(df_array) // 2)
        mape_scores = np.zeros(len(df_array) // 2)
        mae_scores = np.zeros(len(df_array) // 2)
        rmse_scores = np.zeros(len(df_array) // 2)
        
        for i in range(0, len(df_array), 2):
            y_test, y_pred = df_array[i].train_and_predict(df_array[i + 1], target)
            r2_scores[i // 2] = df_array[i].eval_r2(y_test, y_pred)
            mse_scores[i // 2] = df_array[i].eval_mse(y_test, y_pred)
            mape_scores[i // 2] = df_array[i].eval_mape(y_test, y_pred)
            mae_scores[i // 2] = df_array[i].eval_mae(y_test, y_pred)
            rmse_scores[i // 2] = df_array[i].eval_rmse(y_test, y_pred)
            
        r2_scores = pd.Series(r2_scores, index=methods)
        mse_scores = pd.Series(mse_scores, index=methods)
        mape_scores = pd.Series(mape_scores, index=methods)
        mae_scores = pd.Series(mae_scores, index=methods)
        rmse_scores = pd.Series(rmse_scores, index=methods)
        
        return r2_scores, mse_scores, mape_scores, mae_scores, rmse_scores

    def compare_to_org(self, org_train, org_test, df_array, attribute, is_categorical, missing_train, missing_test):
        sim_scores = np.zeros(len(df_array) // 2)
        
        if is_categorical:  
            for i in range(0, len(df_array) - 2, 2):
                imputed_train = df_array[i].loc[missing_train, attribute]
                imputed_test = df_array[i+1].loc[missing_test, attribute]
                
                original_train = org_train.loc[missing_train, attribute]
                original_test = org_test.loc[missing_test, attribute]

                # Calculate categorical similarity (fraction of correctly imputed values)
                sim_train = (imputed_train == original_train).mean()
                sim_test = (imputed_test == original_test).mean()
                
                sim_scores[i // 2] = (sim_train + sim_test) / 2
                
        else:
            # Get attribute range to normalize numerical differences
            min_val = min(org_train[attribute].min(), org_test[attribute].min())
            max_val = max(org_train[attribute].max(), org_test[attribute].max())
            value_range = max_val - min_val if max_val != min_val else 1  # Avoid division by zero
            
            for i in range(0, len(df_array) - 2, 2):            
                imputed_train = df_array[i].loc[missing_train, attribute]
                imputed_test = df_array[i+1].loc[missing_test, attribute]
                
                original_train = org_train.loc[missing_train, attribute]
                original_test = org_test.loc[missing_test, attribute]

                # Compute Mean Absolute Error and normalize it
                mae_train = np.abs(imputed_train - original_train).mean() / value_range
                mae_test = np.abs(imputed_test - original_test).mean() / value_range

                # Convert MAE into a similarity score (higher is better, bound in [0,1])
                sim_scores[i // 2] = 1 - (mae_train + mae_test) / 2
                
        sim_scores[-1] = np.nan # Drop method has no imputed values
        return sim_scores

    def print_scores(self, r2_scores, mse_scores, mape_scores, mae_scores, rmse_scores, sim_scores=None):
        mape_scores = np.array(mape_scores) * 100
        mape_scores = np.round(mape_scores, 3)
        mape_scores = mape_scores.astype(str) + '%'
        
        methods = ['Random', 'Mean', 'Median', 'Frequent', 'KNN', 'Linear Regression', 'Drop']
        scores = pd.DataFrame({
            'R2 Score': np.round(r2_scores, 3),
            'MSE Score': np.round(mse_scores, 3),
            'RMSE Score': np.round(rmse_scores, 3),
            'MAPE Score': mape_scores,
            'MAE Score': np.round(mae_scores, 3),
            'Similarity Score': sim_scores
        }, index=methods)
        
        if sim_scores is None:
            scores.drop(columns=['Similarity Score'], inplace=True)
        else:
            scores['Similarity Score'] = np.round(scores['Similarity Score'], 3)
        
        print("\033[1mError Evaluation Metrics for Different Methods:\033[0m")
        print(scores)

# Example usage
if __name__ == "__main__":
    # Load the data
    df_train = CustomDataFrame(pd.read_csv(r"Final_Project\data\dataset_movies_test.csv"))
    df_test = CustomDataFrame(pd.read_csv(r"Final_Project\data\dataset_movies_test.csv"))

    # Create missing values in the data
    attributes = ['budget', 'revenue']  # Example attributes
    percentages = [10, 20]  # Example percentages of missing values
    df_train, df_test, missing_train, missing_test = df_train.create_mcar(df_test, attributes, percentages)

    # Example of using the custom methods
    df_train.tips()
    df_train.show_nulls(df_test)
    df_train.fill_with_mean(df_test, 'budget')
    df_train.show_nulls(df_test)