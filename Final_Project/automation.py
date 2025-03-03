import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer

import shap
import category_encoders as ce
from scipy.stats import skew
from statsmodels.api import OLS, add_constant
from sklearn.preprocessing import PowerTransformer

""" Loading the data """
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

""" Saving the data """
def save(path, df, index=False):
    df.to_csv(path, index=index)
    print(f"Data saved to {path}")
    return

""" Explore the nulls in the data """
def show_nulls(df_train, df_test):        
    print('Train data:')
    train_nulls = df_train.isnull().sum()
    train_nulls_percentage = (train_nulls / len(df_train)) * 100
    train_nulls_percentage = train_nulls_percentage.round(3)
    train_nulls_percentage = train_nulls_percentage.astype(str) + '%'
    train_nulls_info = pd.concat([train_nulls[train_nulls > 0], train_nulls_percentage[train_nulls > 0]], axis=1)
    train_nulls_info.columns = ['Null Count', 'Percentage']
    print(train_nulls_info)
    
    print()
    print('Test data:')
    test_nulls = df_test.isnull().sum()
    test_nulls_percentage = (test_nulls / len(df_test)) * 100
    test_nulls_percentage = test_nulls_percentage.round(3)
    test_nulls_percentage = test_nulls_percentage.astype(str) + '%'
    test_nulls_info = pd.concat([test_nulls[test_nulls > 0], test_nulls_percentage[test_nulls > 0]], axis=1)
    test_nulls_info.columns = ['Null Count', 'Percentage']
    print(test_nulls_info)
    
    return

""" Artificially creating missing values """

# Create missing values in the data using the Missing Completely at Random (MCAR) mechanism
def create_mcar(df_train, df_test, attributes, percentages):    
    for attribute, percentage in zip(attributes, percentages):
        percentage = float(percentage) / 100  # Convert percentage to a fraction
        
         # Randomly select rows to set as NaN
        missing_train = np.random.choice(df_train.index, 
                                         size=int(percentage * len(df_train)), 
                                         replace=False)
        missing_test = np.random.choice(df_test.index, 
                                        size=int(percentage * len(df_test)), 
                                        replace=False)
        
        df_train.loc[missing_train, attribute] = np.nan
        df_test.loc[missing_test, attribute] = np.nan

    return df_train, df_test, missing_train, missing_test

""" Dealing with missing values """

# Drop the rows with missing values
def drop_missing_values(df_train, df_test, attribute):
    df_train.dropna(subset=[attribute], inplace=True)
    df_test.dropna(subset=[attribute], inplace=True)
    
    return

# Fill missing values with a random value between the minimum and maximum values of the attribute
def fill_randomly(df_train, df_test, attribute):
    min_val = df_train[attribute].min()
    max_val = df_train[attribute].max()
    
    df_train[attribute] = df_train[attribute].fillna(np.random.randint(min_val, max_val))
    df_test[attribute] = df_test[attribute].fillna(np.random.randint(min_val, max_val))
    
    return

# Fill missing values with the mean value of the attribute
def fill_with_mean(df_train, df_test, attribute):
    mean_val = df_train[attribute].mean()
    
    df_train[attribute] = df_train[attribute].fillna(mean_val)
    df_test[attribute] = df_test[attribute].fillna(mean_val)
    
    return

# Fill missing values with the median value of the attribute
def fill_with_median(df_train, df_test, attribute):
    median_val = df_train[attribute].median()
    
    df_train[attribute] = df_train[attribute].fillna(median_val)
    df_test[attribute] = df_test[attribute].fillna(median_val)
    
    return

# Fill missing values with the most frequent value of the attribute
def fill_with_freq(df_train, df_test, attribute):
    freq_val = df_train[attribute].mode()[0] # mode returns a series, so we need to get the first value
    
    df_train[attribute] = df_train[attribute].fillna(freq_val)
    df_test[attribute] = df_test[attribute].fillna(freq_val)
    
    return

# Fill missing values using KNN imputation
def fill_with_knn(df_train, df_test, attribute, is_categorical):
    if is_categorical:
        # Convert categorical data to numerical data using OrdinalEncoder
        encoder = OrdinalEncoder()
        df_train[attribute] = encoder.fit_transform(df_train[[attribute]])
        df_test[attribute] = encoder.transform(df_test[[attribute]])
    
    imputer = KNNImputer(n_neighbors=3, weights='uniform', metric='nan_euclidean')
    
    df_train[attribute] = imputer.fit_transform(df_train[[attribute]])
    df_test[attribute] = imputer.transform(df_test[[attribute]])
    
    if is_categorical:
        # Convert numerical data back to categorical data
        df_train[attribute] = encoder.inverse_transform(df_train[[attribute]])
        df_test[attribute] = encoder.inverse_transform(df_test[[attribute]])
    
    return

# Fill missing values using linear regression
def fill_with_linear_regression(df_train, df_test, attribute, target):
    df_train = pd.get_dummies(df_train, drop_first=True)
    df_test = pd.get_dummies(df_test, drop_first=True)
    
    # Split the data into two sets: one with missing values and one without
    df_train_missing = df_train[df_train[attribute].isnull()]
    df_train_not_missing = df_train.dropna(subset=[attribute])
    
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
    df_train.loc[df_train[attribute].isnull(), attribute] = y_pred
    
    # Repeat the process for the test data
    df_test_missing = df_test[df_test[attribute].isnull()]
    df_test_not_missing = df_test.dropna(subset=[attribute])
    
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
    
    return

""" Training the model and making predictions """
def train_and_predict(df_train, df_test, target):
    X_train = df_train.drop(columns=target)
    encoder = ce.OrdinalEncoder()
    # If there are any categorical columns, convert them to numerical using one-hot encoding
    if df_train.select_dtypes(include='object').shape[1] > 0:
        X_train = encoder.fit_transform(X_train)
    X_train = X_train.fillna(-1)
    
    y_train = df_train[target]
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

""" Evaluating the model """

# R2 Score
def eval_r2(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    return r2

# Mean Squared Error
def eval_mse(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Mean Absolute Percentage Error
def eval_mape(y_test, y_pred):
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mape

# Mean Absolute Error
def eval_mae(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    return mae

# Root Mean Squared Error
def eval_rmse(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

""" Imputing missing values using different methods """

def try_all_methods(df_train, df_test, attribute, target, is_categorical):
    df_train_lr = df_train.copy()
    df_test_lr = df_test.copy()
    fill_with_linear_regression(df_train_lr, df_test_lr, attribute, target)
    
    df_train_r = df_train.copy()
    df_test_r = df_test.copy()
    fill_randomly(df_train_r, df_test_r, attribute)
    
    df_train_mean = df_train.copy()
    df_test_mean = df_test.copy()
    fill_with_mean(df_train_mean, df_test_mean, attribute)
    
    df_train_median = df_train.copy()
    df_test_median = df_test.copy()
    fill_with_median(df_train_median, df_test_median, attribute)

    df_train_freq = df_train.copy()
    df_test_freq = df_test.copy()
    fill_with_freq(df_train_freq, df_test_freq, attribute)
    
    df_train_knn = df_train.copy()
    df_test_knn = df_test.copy()
    fill_with_knn(df_train_knn, df_test_knn, attribute, is_categorical)
    
    df_train_drop = df_train.copy()
    df_test_drop = df_test.copy()
    drop_missing_values(df_train_drop, df_test_drop, attribute)
    
    df_array = [df_train_r, df_test_r, df_train_mean, df_test_mean, df_train_median, df_test_median, df_train_freq, df_test_freq, df_train_knn, df_test_knn, df_train_lr, df_test_lr, df_train_drop, df_test_drop]
    return df_array

""" Comparing the performance of different methods """

def compare_fills(df_array, target):
    methods = ['Random', 'Mean', 'Median', 'Frequent', 'KNN', 'Linear Regression', 'Drop']
    r2_scores = np.zeros(len(df_array) // 2)
    mse_scores = np.zeros(len(df_array) // 2)
    mape_scores = np.zeros(len(df_array) // 2)
    mae_scores = np.zeros(len(df_array) // 2)
    rmse_scores = np.zeros(len(df_array) // 2)
    
    for i in range(0, len(df_array), 2):
        y_test, y_pred = train_and_predict(df_array[i], df_array[i + 1], target)
        r2_scores[i // 2] = eval_r2(y_test, y_pred)
        mse_scores[i // 2] = eval_mse(y_test, y_pred)
        mape_scores[i // 2] = eval_mape(y_test, y_pred)
        mae_scores[i // 2] = eval_mae(y_test, y_pred)
        rmse_scores[i // 2] = eval_rmse(y_test, y_pred)
        
    r2_scores = pd.Series(r2_scores, index=methods)
    mse_scores = pd.Series(mse_scores, index=methods)
    mape_scores = pd.Series(mape_scores, index=methods)
    mae_scores = pd.Series(mae_scores, index=methods)
    rmse_scores = pd.Series(rmse_scores, index=methods)
    
    return r2_scores, mse_scores, mape_scores, mae_scores, rmse_scores

# Compare the imputed data to the original data
def compare_to_org(org_train, org_test, df_array, attribute, is_categorical, missing_train, missing_test):
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

""" Plotting the scores """

def print_scores(r2_scores, mse_scores, mape_scores, mae_scores, rmse_scores, sim_scores=None):
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
    
# """ Colors """
# class color:
#    PURPLE = '\033[95m'
#    CYAN = '\033[96m'
#    DARKCYAN = '\033[36m'
#    BLUE = '\033[94m'
#    GREEN = '\033[92m'
#    YELLOW = '\033[93m'
#    RED = '\033[91m'
#    BOLD = '\033[1m'
#    UNDERLINE = '\033[4m'
#    END = '\033[0m'

""" Mapping the Methods to Indexes """
class Method:
    RANDOM = 0
    MEAN = 2
    MEDIAN = 4
    FREQUENT = 6
    KNN = 8
    LINEAR_REGRESSION = 10
    DROP = 12
    
""" Return the imputed data according to the chosen method """
def get_imputed_data(df_array, method):
    match method:
        case 1:
            return df_array[Method.RANDOM], df_array[Method.RANDOM + 1]
        case 2:
            return df_array[Method.MEAN], df_array[Method.MEAN + 1]
        case 3:
            return df_array[Method.MEDIAN], df_array[Method.MEDIAN + 1]
        case 4:
            return df_array[Method.FREQUENT], df_array[Method.FREQUENT + 1]
        case 5:
            return df_array[Method.KNN], df_array[Method.KNN + 1]
        case 6:
            return df_array[Method.LINEAR_REGRESSION], df_array[Method.LINEAR_REGRESSION + 1]
        case 7:
            return None, None
    
""" User Interface """
def main():
    # Load the data
    path_train = input("Enter the train data file path: ")
    df_train = load_data(path_train)
    path_test = input("Enter the test data file path: ")
    df_test = load_data(path_test)
    
    org_train = df_train.copy()
    org_test = df_test.copy()
    
    # Create missing values in the data
    is_mcar = input("Do you wish to generate missing values in the data? (yes/no): ")
    if is_mcar == 'yes':
        attributes = input("Enter the attributes to generate missing values (separated by commas): ").split(',')
        percentages = input("Enter the percentages of missing values to generate (separated by commas): ").split(',')
        df_train, df_test, missing_train, missing_test = create_mcar(df_train, df_test, attributes, percentages) # Create missing values in the data
     
    # Explore the nulls in the data
    print("\nNull values in the data:")
    show_nulls(df_train, df_test)
    
    print("""\nNow you can choose an attribute to impute.\nHowever, if you choose an attribute that has a low percentage of null values, the imputations method won't improve the scores by much.\nIf you want to stop the process, enter "stop" when asked to enter an attribute.\n""")
    
    tr_and_tst_per_attr = []
    
    while True:
        # Choose the attribute to impute
        attribute = input("Enter the attribute to impute: ")
        
        # If the user wants to stop
        if attribute == 'stop':
            break
        
        # Check if the attribute is categorical
        is_categorical = False
        if df_train[attribute].dtype == 'object':
            is_categorical = True
        
        # Choose the target attribute
        target = input("Enter the target attribute: ")
        
        # Impute the missing values using different methods
        df_array = try_all_methods(df_train, df_test, attribute, target, is_categorical)
        
        # Compare the performance of different methods
        r2_scores, mse_scores, mape_scores, mae_scores, rmse_scores = compare_fills(df_array, target)
        
        # If we used MCAR to generate missing data, we should also evaluate the imputed data comppared to the original
        sim_scores = None
        if is_mcar == 'yes':
            sim_scores = compare_to_org(org_train, org_test, df_array, attribute, is_categorical, missing_train, missing_test)
        
        # Print the scores
        print_scores(r2_scores, mse_scores, mape_scores, mae_scores, rmse_scores, sim_scores)
        
        # Provide explanations for the advantages and disadvantages of the different methods
        print("\n\033[1mAdvantages and disadvantages of the different methods to consider:\033[0m")
        print("""\033[1mRandom:\033[0m This was the baseline method. It is simple and fast, but it does not take into account the relationships between the attributes and the distribution of the data.\nIf you have a large dataset and the missing values are not significant, this method can be used.\nAlso, if the results of the other methods are not significantly better, this method can be used as a fallback.
              
\033[1mMean:\033[0m This method is simple and fast. It is useful when the data is normally distributed and the missing values are not significant. However, it can cause some distortion in the data. 
Let's consider an example of cars dataset and say the horsepower attribute has missing values. If we fill the missing values with the mean of the horsepower, it may distort the data because both a luxury car and an economy car will get the same horsepower value.
                
\033[1mMedian:\033[0m This method is similar to the mean method, but it is more robust to outliers. It is useful when the data is skewed and the missing values are not significant. However, it can also cause some distortion in the data. 
                
\033[1mFrequent:\033[0m This method is useful when the data is categorical and the missing values are not significant. It is simple and fast, but it can cause some distortion in the data.
                
\033[1mKNN:\033[0m This method is useful when the data has a complex relationship between the attributes and the missing values are significant. It takes into account the relationships between the attributes and the distribution of the data. However, it is computationally expensive and may not work well with high-dimensional data.
                
\033[1mLinear Regression:\033[0m This method is useful when the data has a linear relationship between the attributes and the missing values are significant. It takes into account the relationships between the attributes and the distribution of the data. However, it may not work well with non-linear data and may cause overfitting of the data.
                
\033[1mDrop:\033[0m This method is useful when the missing values are insignificant and cannot be imputed. It is simple and fast, but it can cause a loss of information. It should be used when the amount of missing values is small or as a last resort.""")
        
        # Ask the user which imputation method they would like to use
        method = int(input("Enter the method you would like to use (1 for Random, 2 for Mean, 3 for Median, 4 for Frequent, 5 for KNN, 6 for LR, 7 for Drop): "))
        tr_and_tst_per_attr.append((attribute, get_imputed_data(df_array, method)))
        
    # Eventually, save the data according to the chosen methods for each attribute
    to_drop = []
    for attr, data in tr_and_tst_per_attr:
        imputed_train = data[0]
        imputed_test = data[1]
        
        # If the user chose the drop method, we need to remove the samples with null values in the attribute
        if imputed_train is None:
            to_drop.append(attr)
            continue
            
        df_train[attr] = imputed_train[attr]
        df_test[attr] = imputed_test[attr]
        
    # Drop the null values of the attributes in to_drop
    if len(to_drop) > 0:
        df_train.dropna(subset=to_drop, inplace=True)
        df_test.dropna(subset=to_drop, inplace=True)
        
    path = input("Enter the path to save the data: ")
    save(path, df_train)
    save(path, df_test)
    
if __name__=="__main__":
    main()