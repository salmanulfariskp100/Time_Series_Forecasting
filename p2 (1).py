#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# To impute missing Values
from sklearn.impute import SimpleImputer


# In[3]:


data = pd.read_csv("C:\\Users\\govin\\Downloads\\Myocardial infarction complications.csv")


# In[4]:


data.head()


# In[5]:


data.shape


# In[53]:


data_numerics_only = data.select_dtypes(include=np.number)
data_numerics_only


# In[55]:


data_numerics_only.shape


# In[57]:


colnames_numerics_only = data.select_dtypes(include=np.number).columns.tolist()
len(colnames_numerics_only)


# In[16]:


data.dtypes


# In[17]:


data.dtypes.value_counts()


# In[305]:


#printing only 70 columns with highest percentage of Null values
display(round((data.isnull().sum() / (len(data.index)) * 100) , 2).sort_values(ascending = False).head().to_frame().rename({0:'percentage of null values'}, axis = 1).style.background_gradient('magma_r'))
print()
missing = (data.isnull().sum() / (len(data.index)) * 100).to_frame().reset_index().rename({0:'percentage of null values'}, axis = 1)
#ax = sns.barplot(missing['index'],missing['%age'], palette  = 'magma_r')
#ax = sns.barplot(x='index', y='%age', data=missing, palette='magma_r')
#plt.title("Percentage of Missing Values", fontsize = 20)
#plt.xticks(fontsize =7, rotation = 90)
#plt.xlabel("Variables")
#plt.ylabel("Percentage of Missing Values")
#plt.show();


# In[39]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set figure size
plt.figure(figsize=(25, 6))

# Create bar plot
ax = sns.barplot(x='index', y='%age', data=missing, palette='magma_r')

# Set title and axis labels
ax.set_title("Percentage of Missing Values", fontsize=20)
ax.set_xlabel("Variables", fontsize=12)
ax.set_ylabel("Percentage of Missing Values", fontsize=12)

# Rotate x-axis labels
plt.xticks(rotation=90)

# Display plot
plt.show()


# In[169]:


# Heatmap
sns.heatmap(data.isnull(),yticklabels = False, cbar = False,cmap = 'tab20c_r')
plt.title('Missing Data')
plt.show()


# In[239]:


# Total Missing Values of each col in percent
data.isnull().sum()


# In[13]:


# Total Missing Values of each col in percent
(data.isnull().sum()/1700)*100


# In[7]:


# Total Missing Values of complete data
data.isnull().sum().sum()


# In[22]:


# Total Missing Values of complete data in percent
(data.isnull().sum().sum()/(1700*124))*100


# In[25]:


#null values in number >510
null_counts=data.isnull().sum()
null_counts[null_counts>510]


# In[27]:


#null values in percentage >30%
null_counts=(data.isnull().sum()/1700)*100
null_counts[null_counts>30]


# In[29]:


df=data.drop(columns=['IBS_NASL', 'S_AD_KBRIG','D_AD_KBRIG','KFK_BLOOD','NA_KB','NOT_NA_KB','LID_KB'])
df


# In[32]:


df.dtypes.value_counts()


# In[35]:


df.shape


# In[48]:


df.isnull().sum()


# In[49]:


df.isnull().sum().sum()


# In[52]:


#null values in percentage >30%
null_counts=(df.isnull().sum()/1700)*100
len(null_counts[null_counts>30])


# In[34]:


df.describe()


# In[36]:


df.corr()


# In[43]:


# select the float columns
df_float = data.select_dtypes (include= [np.float]) 
df_float


# In[44]:


# select the float columns
df_int = data.select_dtypes (include= [np.int]) 
df_int


# In[ ]:





# In[81]:


imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df))

# Imputation removed column names hence, getting back the names
df_imputed.columns = df.columns

# Checking the final dataset for missing values again.
df_imputed.isnull().sum().sort_values(ascending = False).to_frame().reset_index().rename({0:'Count'}, axis =1).style.background_gradient('magma_r')


# In[82]:


df_imputed


# In[83]:


df_imputed.isnull().sum()


# In[84]:


df_imputed.isnull().sum().sum()


# In[86]:


datatypes=data.dtypes
print(datatypes)


# In[87]:


data.dtypes.value_counts () 


# In[248]:


numeric_cols = df_imp._get_numeric_data().columns  # numeric columns
cols = data.columns  # all columns


# In[249]:


categorical_cols = list(set(cols) - set(numeric_cols))


# In[250]:


categorical_cols


# In[251]:


numeric_cols


# In[256]:


len(numeric_cols)


# In[257]:


len(df_imp)


# In[258]:


df_imp.shape


# In[252]:


len(categorical_cols)


# In[255]:


df_imp.dtypes.value_counts()


# In[247]:


len(numeric_cols)


# In[94]:


datatypes=df_imputed.dtypes
print(datatypes)


# In[95]:


df_imputed.dtypes.value_counts () 


# In[96]:


numeric_colsimp = df_imputed._get_numeric_data().columns  # numeric columns
cols_imp = df_imputed.columns  # all columns


# In[97]:


categorical_colsimp = list(set(cols_imp) - set(numeric_colsimp))


# In[98]:


categorical_colsimp


# In[69]:


numeric_colsimp


# In[70]:


len(categorical_colsimp)


# In[71]:


len(numeric_colsimp)


# In[103]:


df_imp=round(df_imputed,0)
df_imp


# In[ ]:





# In[ ]:





# In[104]:


df_imp.corr()[['LET_IS']].T


# In[105]:


df_imp.drop('ID', axis=1, inplace=True)


# In[106]:


df_imp


# In[306]:


df_imp.corr()[['LET_IS']]


# In[309]:


plt.figure(figsize=(15,10))
sns.heatmap(df_imp.corr())
plt.show()


# In[122]:


# Find the pearson correlations matrix
corr = df_imp.corr(method = 'pearson')[['LET_IS']].T
corr


# In[130]:


corr.unstack().min()


# In[128]:


corr.unstack().idxmin()


# In[131]:


corr.unstack().max()


# In[129]:


corr.unstack().idxmax()


# In[136]:


df_imp['LET_IS'].value_counts().nlargest(n=8)


# In[137]:


df_imp['P_IM_STEN'].value_counts().nlargest(n=2)


# In[230]:


#hc=df_imp.corr()['LET_IS'] > 0.05
#hc.head(60)


# In[229]:


#hc.tail(60)


# In[315]:


a=print(f'Number of patients having  are unknown alive are {data.LET_IS.value_counts()[0]} ,Number of patients having cardiogenic shock  are {data.LET_IS.value_counts()[1]} , Number of patients havingpulmonary edema are{data.LET_IS.value_counts()[2]}, Number of patients having myocardial rupture are {data.LET_IS.value_counts()[3]} ,Number of patients having progress of congestive heart failure are {data.LET_IS.value_counts()[4]}, Number of patients having thromboembolism are {data.LET_IS.value_counts()[5]} ,Number of patients having asystole are {data.LET_IS.value_counts()[6]}, Number of patirnts having ventricular fibrillation are {data.LET_IS.value_counts()[7]}.')
plt.figure(figsize=(12,6),facecolor='pink')
ax=plt.axes()
#ax.set_facecolor("green")
p = sns.countplot(data=df_imp, x="LET_IS", palette='pastel')
plt.legend([['unknown (alive)','cardiogenic shock','pulmonary edema','myocardial rupture','progress of congestive heart failure','thromboembolism','asystole','ventricular fibrillation']])


# In[ ]:





# In[319]:


df_imp['INF_ANAM'].value_counts(normalize=True).plot.bar(title='INF_ANAM')
plt.show()
df_imp['STENOK_AN'].value_counts(normalize=True).plot.bar(title='STENOK_AN')
plt.show()
df_imp['FK_STENOK'].value_counts(normalize=True).plot.bar(title='FK_STENOK')
plt.show()
df_imp['IBS_POST'].value_counts(normalize=True).plot.bar(title='IBS_POST')
plt.show()
df_imp['GB'].value_counts(normalize=True).plot.bar(title='GB')
plt.show()
df_imp['DLIT_AG'].value_counts(normalize=True).plot.bar(title='DLIT_AG')
plt.show()
df_imp['ant_im'].value_counts(normalize=True).plot.bar(title='ant_im')
plt.show()
df_imp['lat_im'].value_counts(normalize=True).plot.bar(title='lat_im')
plt.show()
df_imp['post_im'].value_counts(normalize=True).plot.bar(title='post_im')
plt.show()
df_imp['TIME_B_S'].value_counts(normalize=True).plot.bar(title='TIME_B_S')
plt.show()
df_imp['R_AB_1_n'].value_counts(normalize=True).plot.bar(title='R_AB_1_n')
plt.show()
df_imp['R_AB_2_n'].value_counts(normalize=True).plot.bar(title='R_AB_2_n')
plt.show()
df_imp['R_AB_3_n'].value_counts(normalize=True).plot.bar(title='R_AB_3_n')
plt.show()
df_imp['NA_R_1_n'].value_counts(normalize=True).plot.bar(title='NA_R_1_n')
plt.show()
df_imp['NA_R_2_n'].value_counts(normalize=True).plot.bar(title='NA_R_2_n')
plt.show()
df_imp['NA_R_3_n'].value_counts(normalize=True).plot.bar(title='NA_R_3_n')
plt.show()
df_imp['NOT_NA_1_n'].value_counts(normalize=True).plot.bar(title='NOT_NA_1_n')
plt.show()
df_imp['NOT_NA_2_n'].value_counts(normalize=True).plot.bar(title='NOT_NA_2_n')
plt.show()
df_imp['NOT_NA_3_n'].value_counts(normalize=True).plot.bar(title='NOT_NA_3_n')
plt.show()


# In[235]:


sns.distplot(df_imp['AGE'])
plt.show()
df_imputed['AGE'].plot.box(figsize=(16,5))
plt.show()


# In[106]:


sns.distplot(df_imputed['K_BLOOD'])
plt.show()
df_imputed['K_BLOOD'].plot.box(figsize=(16,5))
plt.show()


# In[107]:


sns.distplot(df_imputed['ALT_BLOOD'])
plt.show()
df_imputed['ALT_BLOOD'].plot.box(figsize=(16,5))
plt.show()


# In[108]:


sns.distplot(df_imputed['AST_BLOOD'])
plt.show()
df_imputed['AST_BLOOD'].plot.box(figsize=(16,5))
plt.show()


# In[109]:


sns.distplot(df_imputed['KFK_BLOOD'])
plt.show()
df_imputed['KFK_BLOOD'].plot.box(figsize=(16,5))
plt.show()


# In[110]:


sns.distplot(df_imputed['L_BLOOD'])
plt.show()
df_imputed['L_BLOOD'].plot.box(figsize=(16,5))
plt.show()


# In[57]:


sns.distplot(df_imputed['ROE'])
plt.show()
df_imputed['ROE'].plot.box(figsize=(16,5))
plt.show()


# In[236]:


plt.figure(figsize = (10,10))
sns.violinplot(x='SEX',y='AGE',data=data)
sns.swarmplot(x=df_imp['SEX'],y=df_imp['AGE'],hue=df_imp['LET_IS'], palette='pastel')


# In[301]:


ax=plt.axis()
sns.countplot(x='AGE', data=df_imp)
plt.figure(figsize=(50,6))


# In[259]:


ax=plt.axis()
sns.countplot(x='SEX', data=df_imp, palette='pastel')
plt.figure(figsize=(50,6))


# In[260]:


ax=plt.axis()
sns.countplot(x='INF_ANAM', data=df_imp, palette='pastel')
plt.figure(figsize=(50,6))


# In[261]:


ax=plt.axis()
sns.countplot(x='STENOK_AN', data=df_imp, palette='pastel')
plt.figure(figsize=(50,6))


# In[262]:


ax=plt.axis()
sns.countplot(x='FK_STENOK', data=df_imp, palette='pastel')
plt.figure(figsize=(50,6))


# In[263]:


ax=plt.axis()
sns.countplot(x='IBS_POST', data=df_imp, palette='pastel')
plt.figure(figsize=(50,6))


# In[266]:


ax=plt.axis()
sns.countplot(x='GB', data=df_imp, palette='pastel')
plt.figure(figsize=(50,6))


# In[267]:


ax=plt.axis()
sns.countplot(x='SIM_GIPERT', data=df_imp, palette='pastel')
plt.figure(figsize=(50,6))


# In[268]:


ax=plt.axis()
sns.countplot(x='DLIT_AG', data=df_imp, palette='pastel')
plt.figure(figsize=(50,6))


# In[270]:


ax=plt.axis()
sns.countplot(x='endocr_01', data=df_imp, palette='pastel')
plt.figure(figsize=(50,6))


# In[271]:


ax=plt.axis()
sns.countplot(x='zab_leg_01', data=df_imp, palette='pastel')
plt.figure(figsize=(50,6))


# In[273]:


ax=plt.axis()
sns.countplot(x='K_SH_POST', data=df_imp, palette='pastel')
plt.figure(figsize=(50,6))


# In[274]:


ax=plt.axis()
sns.countplot(x='ant_im', data=df_imp, palette='pastel')
plt.figure(figsize=(50,6))


# In[ ]:





# In[347]:


SIM_GIPERT=pd.crosstab(df_imp[''],df_imputed['LET_IS'])

SIM_GIPERT.div(SIM_GIPERT.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[232]:


SIM_GIPERT=pd.crosstab(df_imp['SIM_GIPERT'],df_imputed['LET_IS'])
SIM_GIPERT.div(SIM_GIPERT.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[112]:


endocr_01=pd.crosstab(df_imputed['endocr_01'],df_imputed['LET_IS'])
endocr_01.div(endocr_01.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[113]:


zab_leg_01=pd.crosstab(df_imputed['zab_leg_01'],df_imputed['LET_IS'])
zab_leg_01.div(zab_leg_01.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[114]:


O_L_POST=pd.crosstab(df_imputed['O_L_POST'],df_imputed['LET_IS'])
O_L_POST.div(O_L_POST.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[115]:


K_SH_POST=pd.crosstab(df_imputed['K_SH_POST'],df_imputed['LET_IS'])
K_SH_POST.div(K_SH_POST.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[116]:


SVT_POST=pd.crosstab(df_imputed['SVT_POST'],df_imputed['LET_IS'])
SVT_POST.div(SVT_POST.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[117]:


GT_POST=pd.crosstab(df_imputed['GT_POST'],df_imputed['LET_IS'])
GT_POST.div(GT_POST.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[118]:


IM_PG_P=pd.crosstab(df_imputed['IM_PG_P'],df_imputed['LET_IS'])
IM_PG_P.div(IM_PG_P.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[119]:


ritm_ecg_p_01=pd.crosstab(df_imputed['ritm_ecg_p_01'],df_imputed['LET_IS'])
ritm_ecg_p_01.div(ritm_ecg_p_01.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[120]:


n_r_ecg_p_01=pd.crosstab(df_imputed['n_r_ecg_p_01'],df_imputed['LET_IS'])
n_r_ecg_p_01.div(n_r_ecg_p_01.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[121]:


n_p_ecg_p_01=pd.crosstab(df_imputed['n_p_ecg_p_01'],df_imputed['LET_IS'])
n_p_ecg_p_01.div(n_p_ecg_p_01.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[122]:


fibr_ter_01=pd.crosstab(df_imputed['fibr_ter_01'],df_imputed['LET_IS'])
fibr_ter_01.div(fibr_ter_01.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[123]:


GIPO_K=pd.crosstab(df_imputed['GIPO_K'],df_imputed['LET_IS'])
GIPO_K.div(GIPO_K.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[124]:


NOT_NA_KB=pd.crosstab(df_imputed['NOT_NA_KB'],df_imputed['LET_IS'])
NOT_NA_KB.div(NOT_NA_KB.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[125]:


LID_KB=pd.crosstab(df_imputed['LID_KB'],df_imputed['LET_IS'])
LID_KB.div(LID_KB.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[126]:


NITR_S=pd.crosstab(df_imputed['NITR_S'],df_imputed['LET_IS'])
NITR_S.div(NITR_S.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(6,5))
plt.show()


# In[279]:


plt.boxplot(df_imp['AGE'],vert=False)
plt.xlabel("Age")
plt.ylabel("Boxplot")
plt.title("AGE BOXPLOT")


# In[282]:


#plt.boxplot(df_imp['LET_IS'],vert=False)


# In[290]:


sns.scatterplot(df_imp['AGE'])


# In[300]:


df_imp.LET_IS.value_counts().plot(kind="pie")


# In[298]:


def f(LET_IS):
    if LET_IS == 0:
        return 'unknown (alive)'
    elif LET_IS== 1:
        return 'cardiogenic shock'
    elif LET_IS == 2:
        return 'pulmonary edema'
    elif LET_IS ==3:
        return 'myocardial rupture'
    elif LET_IS ==4:
        return 'progress of congestive heart failure'
    elif LET_IS ==5:
        return 'thromboembolism'
    elif LET_IS ==6:
        return 'asystole'
    else:
        return 'ventricular fibrillation'


df_imp['LET_IS_cat'] = df_imp['LET_IS'].apply(f)


# In[299]:


df_imp


# In[346]:


a=print(f'Number of patients having  are unknown alive are {data.LET_IS.value_counts()[0]} ,Number of patients having cardiogenic shock  are {data.LET_IS.value_counts()[1]} , Number of patients havingpulmonary edema are{data.LET_IS.value_counts()[2]}, Number of patients having myocardial rupture are {data.LET_IS.value_counts()[3]} ,Number of patients having progress of congestive heart failure are {data.LET_IS.value_counts()[4]}, Number of patients having thromboembolism are {data.LET_IS.value_counts()[5]} ,Number of patients having asystole are {data.LET_IS.value_counts()[6]}, Number of patirnts having ventricular fibrillation are {data.LET_IS.value_counts()[7]}.')
plt.figure(figsize=(12,6),facecolor='pink')
ax=plt.axes()
b=['unknown(alive)','cardiogenicshock','pulmonaryedema','myocardialrupture','progressofcongestiveheartfailure','thromboembolism','asystole','ventricularfibrillation']
ax.set_xticklabels(b, rotation=90)
p = sns.countplot(data=df_imp, x="LET_IS_cat", palette='pastel')


# In[ ]:




