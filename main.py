
# Link : https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing
# Import essential libraries
import pandas as pd
import scipy.stats as st
import seaborn as sns
import numpy as np
import matplotlib
from scipy.stats import mannwhitneyu
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

# Plot 1
# Insights: Almost 50% to 50% in total by Groups
def version_pie_portion(dataframe, col,target):
    plt.figure(figsize=(10,5), dpi = 100)
    target_df = dataframe.groupby([col])[target].agg(['count']).reset_index()

    plt.pie(target_df['count'],labels = target_df[col],
            autopct='%1.2f%%', startangle=45, colors=sns.color_palette('Set1'),
            labeldistance=0.55, pctdistance=0.2)
    plt.title('Version Portion in Total', fontsize = 20)
    plt.axis('off')
    plt.legend()
    plt.show()


def box_plot(df,target = 'retention_1'):
    #version_df = df.groupby(['version',target])['sum_gamerounds'].agg(['sum','median']).reset_index()
    sns.catplot(df, x = 'version', y = 'sum_gamerounds', hue = target, kind = 'box')
    plt.show()
    #fig, axs = plt.subplots(1,2, figsize = (15,10))
    #print(version_df.head())


def distribution_plt(dataframe,column_name,title,xlabel,ylabel):
    # Distribution of Game Rounds by User ID
    sns.histplot(dataframe[column_name], color = 'red')
    plt.title(title, fontsize = 30)
    plt.xlabel(xlabel, fontsize = 15)
    plt.ylabel(ylabel)
    plt.axvline(np.median(dataframe[column_name]), 0, linestyle='--', linewidth=1.5, color='b')
    plt.show()

# Plot 2
# Insights: Left Skewed Diagram
def user_rounds(df):
    gamerounds_userid = df.groupby(['sum_gamerounds'])['userid'].count().reset_index()
    gamerounds_userid.columns = ['Rounds', '# of User']
    distribution_plt(gamerounds_userid, 'Rounds','# of User by Game Rounds', 'Rounds','# of User')
    #print(gamerounds_userid.head())
    # return gamerounds_userid


def AB_test(df, target = 'retention_1'):
    sub_df = df[['userid','version','sum_gamerounds',target]].copy(deep = True)
    # 1. Establish the Hypothesis:
    #*  H0 :  Gate_30 == Gate _40
    #*  H1 :  Gate_30 != Gate_40
    # H0 : There is no statistcal difference between gate_30 && gate_40

    # 2. Hypothesis Implementation
    # i) If the distribution is normal & variance are homogeneous, T-test are applied
    # ii) If the distribution is normal & variances are not homogeneous, Welch Test are to be used.
    # iii) If the distribution is not normal & variances are not homogeneous, Mann Whitney U
    # Test directly (non-parametric test) are to be used.

    # Via Previous Visualisation and Insights, the data set is not normally distributed.
    # Thus, Mann Whitney U Test directly used.
    pvalue = mannwhitneyu(sub_df.loc[sub_df['version'] == 'gate_30', 'sum_gamerounds'],
                          sub_df.loc[sub_df['version'] == 'gate_40', 'sum_gamerounds'])
    #print(type(pvalue))
    #print(pvalue)

    # 3. Interpret the result
    if (pvalue[1] < 0.05):
        print('Mann Whitney U Test Result \n')
        print("H0 Hypothesis is Rejected. That is, there is a statistically signficiant difference between them")
    else:
        print('Mann Whitney U Test Result \n')
        print("H0 Hypothesis is Not Rejected. That is, there is no statistically signficiant difference between them")

    # Method 2
    version_df = df.groupby('version')['retention_1'].agg(['median']).reset_index()
    print(version_df.head())
    # It comes up with mean difference between two version

    # Am I confident in such difference -- bootstrapping : re-sample the dataset with replacement and
    # calculate target column retention for those samples. The variation in target column will give us
    # an indication of how uncertain the retention numbers are
    bootstrapping = []
    for i in range(1000):
        boot_mean = df.sample(frac = 1, replace = True).groupby('version')['retention_1'].mean()
        bootstrapping.append(boot_mean)

    variation = pd.DataFrame(bootstrapping)
    variation['diff'] = (variation['gate_30'] - variation['gate_40']) / variation['gate_40'] * 100

    # Generate the density plot to show distribution of difference between two groups
    variation['diff'].plot(kind = 'density')
    plt.show()
    plt.savefig('output/Density plot.png')

    print(f'Probability that {target} retention is higher when (gate 30) is :',
          (variation['diff'] > 0).mean())

    # print(variation.head(20))
    return variation,version_df,pvalue


if __name__ == '__main__':
    cats = data_exploration()
    print('Basic Information about Data Set')
    print('-' * 100)
    df_exploration(cats)
    print('-' * 100)
    #version_pie_portion(cats, 'version','userid')
    #user_rounds(cats)
    # box_plot(cats)
    variation,version_df,pvalue = AB_test(cats)
