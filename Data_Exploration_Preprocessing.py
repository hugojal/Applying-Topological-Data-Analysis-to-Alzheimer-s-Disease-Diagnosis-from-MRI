df.head()

df = df.loc[df['Visit']==1]

df = df.reset_index(drop=True)

df['M/F'] = df['M/F'].replace(['F','M'], [0,1])

df['Group'] = df['Group'].replace(['Converted'], ['Demented'])

df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1,0])

df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1)

def bar_chart(feature):
    Demented = df[df['Group']==1][feature].value_counts()
    Nondemented = df[df['Group']==0][feature].value_counts()

    df_bar = pd.DataFrame([Demented, Nondemented])
    df_bar.index = ['Demented', 'Nondemented']

    df_bar.plot(kind='bar', stacked=True, figsize=(8, 5))

import matplotlib.pyplot as plt


bar_chart('M/F')


plt.xlabel('Group')


plt.ylabel('Number of patients')




plt.legend()


plt.title('Gender and Demented rate')

import seaborn as sns


facet = sns.FacetGrid(df, hue="Group", aspect=3)


facet.map(sns.kdeplot, 'MMSE', shade=True)


facet.set(xlim=(0, df['MMSE'].max()))


facet.add_legend()




plt.xlim(15, 30)

facet = sns.FacetGrid(df, hue="Group", aspect=3)  # FacetGrid
facet.map(sns.kdeplot, 'ASF', shade=True)  # KDE
facet.set(xlim=(0, df['ASF'].max()))
facet.add_legend()
plt.xlim(0.5, 2)

facet = sns.FacetGrid(df, hue="Group", aspect=3)  # FacetGrid
facet.map(sns.kdeplot, 'eTIV', shade=True)  # KDE
facet.set(xlim=(0, df['eTIV'].max()))
facet.add_legend()
plt.xlim(900, 2100)

facet = sns.FacetGrid(df, hue="Group", aspect=3)  # FacetGrid , 'Group'
facet.map(sns.kdeplot, 'nWBV', shade=True)  # KDE
facet.set(xlim=(0, df['nWBV'].max()))
facet.add_legend()
plt.xlim(0.6, 0.9)

facet = sns.FacetGrid(df, hue="Group", aspect=3)  # FacetGrid, 'Group'에
facet.map(sns.kdeplot, 'Age', shade=True)  # KDE
facet.set(xlim=(0, df['Age'].max()))
facet.add_legend()
plt.xlim(50, 100)

facet = sns.FacetGrid(df, hue="Group", aspect=3)  # FacetGrid, 'Group'
facet.map(sns.kdeplot, 'EDUC', shade=True)  # KDE
facet.set(xlim=(df['EDUC'].min(), df['EDUC'].max()))
facet.add_legend()
plt.ylim(0, 0.16)

missing_values = pd.isnull(df).sum()




missing_values

df_dropna = df.dropna(axis=0, how='any')
missing_values_after_drop = pd.isnull(df_dropna).sum()


missing_values_after_drop

group_value_counts = df_dropna['Group'].value_counts()
# 'Group'


# 'Group'
group_value_counts

educ_ses_median = df.groupby(['EDUC'])['SES'].median()


educ_ses_median

df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"), inplace=True)
missing_values_check = pd.isnull(df['SES']).value_counts()


missing_values_check
