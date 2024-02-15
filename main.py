
# Link : https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing
# Import essential libraries
import pandas as pd
import scipy.stats as st
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def df_exploration(df):
    """
        Information about the df
        # Columns Data Type
        # Data Frame shape
        # Columns Name
        # Columns Description
        # First 5 Data Samples
    """
    features_dtypes = df.dtypes
    rows, columns = df.shape

    missing_values_cols = df.isnull().sum()
    missing_col = missing_values_cols.sort_values(ascending=False)
    features_names = missing_col.index.values
    missing_values = missing_col.values

    print('=' * 50)
    print('===> This data frame contains {} rows and {} columns'.format(rows, columns))
    print('=' * 50)

    print("{:13}{:13}{:30}".format('Feature Name'.upper(),
                                   'Data Format'.upper(),
                                   'The first five samples'.upper()))

    for features_names, features_dtypes in zip(features_names, features_dtypes[features_names]):
        print('{:15} {:14} '.format(features_names, str(features_dtypes)), end=" ")

        for i in range(5):
            print(df[features_names].iloc[i], end=",")

        print("=" * 50)

def data_exploration():
    cats = pd.read_csv( 'data/cookie_cats.csv')
    # print(cats.head(20))
    return cats

def levene_test(df, col, features):
    """
        The Levene test tests the null hypothesis that all input samples are from populations with equal variances.
        Levene’s test is an alternative to Bartlett’s test bartlett in the case where there are significant deviations from normality.

    """
    group_A = df[df[col] == features[0]].sum_gamerounds
    group_B = df[df[col] == features[1]].sum_gamerounds
    result = st.levene(group_A, group_B)

    return result

def hist_plot(df):
    df_version = df.groupby(['version','retention_1'])['sum_gamerounds'].mean().reset_index()
    # sns.factorplot(data=df_version,kind='count',x='retention_1',col='version')

    #df_version.plot(kind = 'bar')
    ##plt.title('Gamerdouns ')
    #plt.xticks( rotation = 0)
    #plt.show()


def distribution_plt(dataframe,column_name,title,xlabel,ylabel):
    # Distribution of Game Rounds by User ID
    sns.distplot(dataframe[column_name], color = 'red')
    plt.title(title, fontsize = 30)
    plt.xlabel(xlabel, fontsize = 15)
    plt.ylabel(ylabel)
    plt.axvline(np.median(dataframe[column_name]), 0, linestyle='--', linewidth=1.5, color='b')
    plt.show()

def gamerounds_user(df):
    gamerounds_userid = df.groupby(['sum_gamerounds'])['userid'].count().reset_index()
    #gamerounds_userid.columns = ['rounds', ''
    return gamerounds_userid


if __name__ == '__main__':
    cats = data_exploration()
    df_exploration(cats)
    #gate_30, gate_40 = ANOVA_test(cats)
    # distribution_plt(cats, 'sum_gamerounds', title = 'Distribution of Game Rounds', xlabel = '', ylabel='')
    #$ hist_plot(cats)
    temp = gamerounds_user(cats)
    print(temp)