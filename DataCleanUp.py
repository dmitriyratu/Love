# Databricks notebook source
import pandas as pd
import numpy as np

from ydata_profiling import ProfileReport
import missingno as msno

import matplotlib.pyplot as plt
from ipywidgets import interact

from uszipcode import SearchEngine

# COMMAND ----------

# MAGIC %run /Repos/dmitriy.ratu@gmail.com/Love/Config

# COMMAND ----------

# MAGIC %run /Repos/dmitriy.ratu@gmail.com/Love/Functions

# COMMAND ----------

df_raw = pd.read_csv('SpeedDatingData.csv', encoding='ISO-8859-1')

# COMMAND ----------

d_data = {

    'round_info':[
        'wave','round','condtn'
        ],

    'cat_user_prfl':[
        'iid','field','uni','gender','race','from_country','from_continent','zipcode','goal','career','date_3','numdat_2','length'
        ],
    
    'cat_partner':[
        'pid','samerace','dec_o','match'
        ],

    'num_user_prfl':[
        'int_corr','age','income','mn_sat','tuition','imprace','imprelig',
        'sports','tvsports','exercise','dining','museums','art','hiking',
        'gaming','clubbing','reading','tv','theater','movies','concerts',
        'music','shopping','yoga',
        'exphappy','satis_2','expnum','match_es',
        'you_call','them_cal','numdates_3',
        
        ],
    
    'num_partner':[
        'order'
        ],

    'ord_user_prfl':[
        'date','go_out'
        ],

    'num_scales':[

        # ----------------------------- PRE-EVENT ----------------------------- #
        # What do you look for in the opposite sex
        'attr1_1','sinc1_1','intel1_1','fun1_1','amb1_1','shar1_1',
        # What you think MOST of your fellow men/women look for in the opposite sex
        'attr4_1','sinc4_1','intel4_1','fun4_1','amb4_1','shar4_1',
        # What do you think the opposite sex looks for in a date
        'attr2_1','sinc2_1','intel2_1','fun2_1','amb2_1','shar2_1',
        # How do you think you measure up
        'attr3_1','sinc3_1','intel3_1','fun3_1','amb3_1',
        # How do you think others perceive you
        'attr5_1','sinc5_1','intel5_1','fun5_1','amb5_1',

        # ----------------------------- HALF WAY THROUGH EVENT ----------------------------- #
        # What do you look for in the opposite sex
        'attr1_s','sinc1_s','intel1_s','fun1_s','amb1_s','shar1_s',
        # How do you think you measure up
        'attr3_s','sinc3_s','intel3_s','fun3_s','amb3_s',

        # ----------------------------- POST EVENT - PRE-MATCHES ----------------------------- #
        # Distribute the points among these six attributes in the way that best reflects the actual importance of these attributes in your decisions
        'attr7_2','sinc7_2','intel7_2','fun7_2','amb7_2','shar7_2',
        # We want to know what you look for in the opposite sex
        'attr1_2','sinc1_2','intel1_2','fun1_2','amb1_2','shar1_2',
        # What do you think MOST of your fellow gender look for in the opposite sex?
        'attr4_2','sinc4_2','intel4_2','fun4_2','amb4_2','shar4_2',
        # What do you think the opposite sex looks for in a date?
        'attr2_2','sinc2_2','intel2_2','fun2_2','amb2_2','shar2_2',
        # How do you think you measure up?
        'attr3_2','sinc3_2','intel3_2','fun3_2','amb3_2',
        # How do you think others perceive you?
        'attr5_2','sinc5_2','intel5_2','fun5_2','amb5_2',

        # ----------------------------- POST EVENT - POST-MATCHES ----------------------------- #
        # What do you look for in the opposite sex
        'attr1_3','sinc1_3','intel1_3','fun1_3','amb1_3','shar1_3',
        # Distribute the points among these six attributes in the way that best reflects the actual importance of these attributes in your decisions
        'attr7_3','sinc7_3','intel7_3','fun7_3','amb7_3','shar7_3',
        # What do you think MOST of your fellow gender look for in the opposite sex?
        'attr4_3','sinc4_3','intel4_3','fun4_3','amb4_3','shar4_3',
        # What do you think the opposite sex looks for in a date?
        'attr2_3','sinc2_3','intel2_3','fun2_3','amb2_3','shar2_3',
        # How do you think you measure up?
        'attr3_3','sinc3_3','intel3_3','fun3_3','amb3_3',
        # How do you think others perceive you?
        'attr5_3','sinc5_3','intel5_3','fun5_3','amb5_3',
        ],

}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalize Data

# COMMAND ----------

df_raw['zipcode'] = df_raw['zipcode'].str.replace(',','')
df_raw['income'] = df_raw['income'].str.replace(',','').astype(float)
df_raw['numdates_3'] = df_raw[['numdat_3','num_in_3']].max(axis = 1)

# COMMAND ----------

sr = SearchEngine()

INFLATION_COEF = 1.67

zip_code_dict = {}
for i in df_raw['zipcode'].unique():
    try: zip_code_dict[i] = int(sr.by_zipcode(i).to_dict()['median_household_income']/INFLATION_COEF)
    except:continue

df_raw['income'] = df_raw['zipcode'].map(lambda x: zip_code_dict.get(x))

# COMMAND ----------

df_raw.dropna(subset = ['iid','pid'], inplace = True)
df_raw = df_raw[~df_raw[['iid','pid']].isin(df_raw.loc[df_raw['goal'].isna(),'iid'].unique()).any(axis = 1)]

df_raw[['age','age_o']] = df_raw[['age','age_o']].fillna(df_raw['age'].median())

# COMMAND ----------

binary_mapping = {1:'Yes',0:'No',np.nan:'missing'}

df_raw['from_country'] = df_raw['from'].map(from_mapping)
df_raw['from_continent'] = df_raw['from_country'].map(continent_mapping)

df_raw['date_3'] = df_raw['date_3'].map(binary_mapping)

df_raw['uni'] = df_raw['undergra'].map(uni_mapping).fillna('missing')

df_raw['race'] = df_raw['race'].map(race_mapping)
df_raw['goal'] = df_raw['goal'].map(goal_mapping)

df_raw['condtn'] = df_raw['condtn'].map({1:'limited choice',2:'extensive choice',np.nan:'missing'})

df_raw['samerace'] = df_raw['samerace'].map(binary_mapping)
df_raw['dec_o'] = df_raw['dec_o'].map(binary_mapping)
df_raw['match'] = df_raw['match'].map(binary_mapping)

df_raw['date'] = df_raw['date'].max() - df_raw['date']
df_raw['go_out'] = df_raw['go_out'].max() - df_raw['go_out']

df_raw['numdat_2'] = df_raw['numdat_2'].map(numdate_mapping).fillna('missing')
df_raw['length'] = df_raw['length'].map(length_mapping).fillna('missing')

# COMMAND ----------

df_raw['date_3'] = df_raw['date_3'].fillna('missing')
df_raw.loc[(df_raw['date_3'] == 'No') & df_raw['numdates_3'].isna(),'numdates_3'] = 0
df_raw.loc[df_raw['numdates_3'] > 0,'date_3'] = 'Yes'
df_raw.loc[(df_raw['date_3'] == 'Yes') & (df_raw['numdates_3'] == 0), 'numdates_3'] = np.nan

# COMMAND ----------

hobby_cols = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv',  'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga']
df_raw[hobby_cols] = df_raw[hobby_cols].clip(lower = 1, upper = 10)

# COMMAND ----------

d_data.keys()

# COMMAND ----------

disp_all(df_raw[d_data['num_scales']].head())

# COMMAND ----------



# COMMAND ----------

@interact(key = d_data.keys())
def f(key):

    fig, axes = plt.subplots(df_raw['wave'].nunique()//3, 3, figsize=(15, 20))

    dfs = [df_raw.query(f"wave == {wave}")[d_data[key]] for wave in df_raw['wave'].unique()]

    flat_axes = axes.flatten()
    for n, (ax, subset) in enumerate(zip(flat_axes, dfs)):
        msno.matrix(subset, ax=ax, sparkline = False, fontsize = 9)
        ax.set_title(f"Wave: {n + 1}")

    fig.suptitle(key, fontsize = 16, y = 1.05)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

lst = []
d2 = df_raw[d_data['num_scales']].isna().astype(int)
for wave in df_raw['wave'].unique():
    d2.loc[df_raw['wave'] == wave]
    lst.append(d2.loc[df_raw['wave'] == wave].mean().values)

# COMMAND ----------

import plotly.express as px
fig = px.imshow(lst, x = d_data['num_scales'], y = df_raw['wave'].unique())
fig.update_layout(width=900, height=500)
fig.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

profile = ProfileReport(df_raw, minimal = True, title = "Profiling Report")
