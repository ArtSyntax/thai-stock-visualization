#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas_profiling 


# In[3]:


df = pd.read_csv('./price_ratio_cagr_sector.csv')


# Prepare for model
# 

# In[4]:


df.head(5)


# In[5]:


df.columns


# # Select feature

# In[6]:


sf2012 = ['P_201201', 
'EPS_2012',
'GPM_2012',
'Ops_2012',
'EBIT_2012',
'NPM_2012',
'IBD_2012',
'ROE_2012',
'ROA_2012',
'DPS_2012',
'BV_2012',
'Payout_2012']


# In[7]:


sf2013 = ['P_201301', 
'EPS_2013',
'GPM_2013',
'Ops_2013',
'EBIT_2013',
'NPM_2013',
'IBD_2013',
'ROE_2013',
'ROA_2013',
'DPS_2013',
'BV_2013',
'Payout_2013']


# In[8]:


sf2014 = ['P_201401', 
'EPS_2014',
'GPM_2014',
'Ops_2014',
'EBIT_2014',
'NPM_2014',
'IBD_2014',
'ROE_2014',
'ROA_2014',
'DPS_2014',
'BV_2014',
'Payout_2014']


# In[9]:


sf2015 = ['P_201501', 
'EPS_2015',
'GPM_2015',
'Ops_2015',
'EBIT_2015',
'NPM_2015',
'IBD_2015',
'ROE_2015',
'ROA_2015',
'DPS_2015',
'BV_2015',
'Payout_2015']


# In[10]:


sf2016 = ['P_201601', 
'EPS_2016',
'GPM_2016',
'Ops_2016',
'EBIT_2016',
'NPM_2016',
'IBD_2016',
'ROE_2016',
'ROA_2016',
'DPS_2016',
'BV_2016',
'Payout_2016']


# In[11]:


sf2017 = ['P_201701', 
'EPS_2017',
'GPM_2017',
'Ops_2017',
'EBIT_2017',
'NPM_2017',
'IBD_2017',
'ROE_2017',
'ROA_2017',
'DPS_2017',
'BV_2017',
'Payout_2017']


# # Feature correlation

# In[12]:


df[sf2012]


# In[13]:


df[sf2012].corr()


# In[14]:


plt.figure(figsize=(10,10))
plt.matshow(df[sf2012].corr('spearman'),fignum=1,cmap='Blues')
plt.xticks(np.arange(12),df[sf2012].corr().columns,rotation=90)
plt.yticks(np.arange(12),df[sf2012].corr().columns,rotation=0)
plt.colorbar()


# In[15]:


plt.figure(figsize=(10,10))
plt.matshow(df[sf2013].corr('spearman'),fignum=1,cmap='Blues')
plt.xticks(np.arange(12),df[sf2013].corr().columns,rotation=90)
plt.yticks(np.arange(12),df[sf2013].corr().columns,rotation=0)
plt.colorbar()


# In[16]:


plt.figure(figsize=(10,10))
plt.matshow(df[sf2014].corr('spearman'),fignum=1,cmap='Blues')
plt.xticks(np.arange(12),df[sf2014].corr().columns,rotation=90)
plt.yticks(np.arange(12),df[sf2014].corr().columns,rotation=0)
plt.colorbar()


# In[17]:


plt.figure(figsize=(10,10))
plt.matshow(df[sf2015].corr('spearman'),fignum=1,cmap='Blues')
plt.xticks(np.arange(12),df[sf2015].corr().columns,rotation=90)
plt.yticks(np.arange(12),df[sf2015].corr().columns,rotation=0)
plt.colorbar()


# In[18]:


plt.figure(figsize=(10,10))
plt.matshow(df[sf2016].corr('spearman'),fignum=1,cmap='Blues')
plt.xticks(np.arange(12),df[sf2016].corr().columns,rotation=90)
plt.yticks(np.arange(12),df[sf2016].corr().columns,rotation=0)
plt.colorbar()


# In[19]:


plt.figure(figsize=(10,10))
plt.matshow(df[sf2017].corr('spearman'),fignum=1,cmap='Blues')
plt.xticks(np.arange(12),df[sf2017].corr().columns,rotation=90)
plt.yticks(np.arange(12),df[sf2017].corr().columns,rotation=0)
plt.colorbar()


# # Price correlation

# In[20]:


corr2012 = df[sf2012].corr()['P_201201'].rename({
    'P_201201' : 'P', 
    'EPS_2012' : 'EPS', 
    'GPM_2012' : 'GPM', 
    'Ops_2012' : 'Ops', 
    'EBIT_2012' : 'EBIT', 
    'NPM_2012' : 'NPM', 
    'IBD_2012' : 'IBD', 
    'ROE_2012' : 'ROE', 
    'ROA_2012' : 'ROA', 
    'DPS_2012' : 'DPS', 
    'BV_2012' : 'BV', 
    'Payout_2012' : 'Payout'
}) #.sort_values(ascending=False)


# In[21]:


corr2013 = df[sf2013].corr()['P_201301'].rename({
    'P_201301' : 'P', 
    'EPS_2013' : 'EPS', 
    'GPM_2013' : 'GPM', 
    'Ops_2013' : 'Ops', 
    'EBIT_2013' : 'EBIT', 
    'NPM_2013' : 'NPM', 
    'IBD_2013' : 'IBD', 
    'ROE_2013' : 'ROE', 
    'ROA_2013' : 'ROA', 
    'DPS_2013' : 'DPS', 
    'BV_2013' : 'BV', 
    'Payout_2013' : 'Payout'
})


# In[22]:


corr2014 = df[sf2014].corr()['P_201401'].rename({
    'P_201401' : 'P', 
    'EPS_2014' : 'EPS', 
    'GPM_2014' : 'GPM', 
    'Ops_2014' : 'Ops', 
    'EBIT_2014' : 'EBIT', 
    'NPM_2014' : 'NPM', 
    'IBD_2014' : 'IBD', 
    'ROE_2014' : 'ROE', 
    'ROA_2014' : 'ROA', 
    'DPS_2014' : 'DPS', 
    'BV_2014' : 'BV', 
    'Payout_2014' : 'Payout'
})


# In[23]:


corr2015 = df[sf2015].corr()['P_201501'].rename({
    'P_201501' : 'P', 
    'EPS_2015' : 'EPS', 
    'GPM_2015' : 'GPM', 
    'Ops_2015' : 'Ops', 
    'EBIT_2015' : 'EBIT', 
    'NPM_2015' : 'NPM', 
    'IBD_2015' : 'IBD', 
    'ROE_2015' : 'ROE', 
    'ROA_2015' : 'ROA', 
    'DPS_2015' : 'DPS', 
    'BV_2015' : 'BV', 
    'Payout_2015' : 'Payout'
})


# In[24]:


corr2016 = df[sf2016].corr()['P_201601'].rename({
    'P_201601' : 'P', 
    'EPS_2016' : 'EPS', 
    'GPM_2016' : 'GPM', 
    'Ops_2016' : 'Ops', 
    'EBIT_2016' : 'EBIT', 
    'NPM_2016' : 'NPM', 
    'IBD_2016' : 'IBD', 
    'ROE_2016' : 'ROE', 
    'ROA_2016' : 'ROA', 
    'DPS_2016' : 'DPS', 
    'BV_2016' : 'BV', 
    'Payout_2016' : 'Payout'
})


# In[25]:


corr2017 = df[sf2017].corr()['P_201701'].rename({
    'P_201701' : 'P', 
    'EPS_2017' : 'EPS', 
    'GPM_2017' : 'GPM', 
    'Ops_2017' : 'Ops', 
    'EBIT_2017' : 'EBIT', 
    'NPM_2017' : 'NPM', 
    'IBD_2017' : 'IBD', 
    'ROE_2017' : 'ROE', 
    'ROA_2017' : 'ROA', 
    'DPS_2017' : 'DPS', 
    'BV_2017' : 'BV', 
    'Payout_2017' : 'Payout'
})


# In[26]:


corr_mat = pd.DataFrame({
    'corr2012': corr2012,
    'corr2013': corr2013,
    'corr2014': corr2014,
    'corr2015': corr2015,
    'corr2016': corr2016,
    'corr2017': corr2017
})


# In[27]:


corr_mat_sort = corr_mat.sort_values(by='corr2012', ascending=False)
corr_mat_sort


# In[28]:


corr_mat_sort.T


# In[29]:


# plt.rcParams["axes.grid"] = True


# In[30]:


plt.figure(figsize=(12,6))
plt.matshow(corr_mat_sort.T,fignum=1,cmap='Blues')
plt.xticks(np.arange(12), corr_mat_sort.T.columns,rotation=0)
plt.yticks(np.arange(6), corr_mat_sort.T.index,rotation=0)
plt.colorbar()


# # Diff

# ## make diff column

# In[31]:


diff_df = pd.DataFrame({
    'P_2013': df['P_201301'],
    'P_2014': df['P_201401'],
    'P_2015': df['P_201501'],
    'P_2016': df['P_201601'],
    'P_2017': df['P_201701'],

    'P_D_2013_2012':  df['P_201301'] - df['P_201201'], 
    'P_D_2014_2013':  df['P_201401'] - df['P_201301'], 
    'P_D_2015_2014':  df['P_201501'] - df['P_201401'], 
    'P_D_2016_2015':  df['P_201601'] - df['P_201501'], 
    'P_D_2017_2016':  df['P_201701'] - df['P_201601'], 
    
    'EPS_D_2013_2012':  df['EPS_2013'] - df['EPS_2012'], 
    'EPS_D_2014_2013':  df['EPS_2014'] - df['EPS_2013'], 
    'EPS_D_2015_2014':  df['EPS_2015'] - df['EPS_2014'], 
    'EPS_D_2016_2015':  df['EPS_2016'] - df['EPS_2015'], 
    'EPS_D_2017_2016':  df['EPS_2017'] - df['EPS_2016'], 
    
    'GPM_D_2013_2012':  df['GPM_2013'] - df['GPM_2012'], 
    'GPM_D_2014_2013':  df['GPM_2014'] - df['GPM_2013'], 
    'GPM_D_2015_2014':  df['GPM_2015'] - df['GPM_2014'], 
    'GPM_D_2016_2015':  df['GPM_2016'] - df['GPM_2015'], 
    'GPM_D_2017_2016':  df['GPM_2017'] - df['GPM_2016'], 
    
    'Ops_D_2013_2012':  df['Ops_2013'] - df['Ops_2012'], 
    'Ops_D_2014_2013':  df['Ops_2014'] - df['Ops_2013'], 
    'Ops_D_2015_2014':  df['Ops_2015'] - df['Ops_2014'], 
    'Ops_D_2016_2015':  df['Ops_2016'] - df['Ops_2015'], 
    'Ops_D_2017_2016':  df['Ops_2017'] - df['Ops_2016'], 
    
    'EBIT_D_2013_2012':  df['EBIT_2013'] - df['EBIT_2012'], 
    'EBIT_D_2014_2013':  df['EBIT_2014'] - df['EBIT_2013'], 
    'EBIT_D_2015_2014':  df['EBIT_2015'] - df['EBIT_2014'], 
    'EBIT_D_2016_2015':  df['EBIT_2016'] - df['EBIT_2015'], 
    'EBIT_D_2017_2016':  df['EBIT_2017'] - df['EBIT_2016'], 
    
    'NPM_D_2013_2012':  df['NPM_2013'] - df['NPM_2012'], 
    'NPM_D_2014_2013':  df['NPM_2014'] - df['NPM_2013'], 
    'NPM_D_2015_2014':  df['NPM_2015'] - df['NPM_2014'], 
    'NPM_D_2016_2015':  df['NPM_2016'] - df['NPM_2015'], 
    'NPM_D_2017_2016':  df['NPM_2017'] - df['NPM_2016'], 
    
    'IBD_D_2013_2012':  df['IBD_2013'] - df['IBD_2012'], 
    'IBD_D_2014_2013':  df['IBD_2014'] - df['IBD_2013'], 
    'IBD_D_2015_2014':  df['IBD_2015'] - df['IBD_2014'], 
    'IBD_D_2016_2015':  df['IBD_2016'] - df['IBD_2015'], 
    'IBD_D_2017_2016':  df['IBD_2017'] - df['IBD_2016'], 
    
    'ROE_D_2013_2012':  df['ROE_2013'] - df['ROE_2012'], 
    'ROE_D_2014_2013':  df['ROE_2014'] - df['ROE_2013'], 
    'ROE_D_2015_2014':  df['ROE_2015'] - df['ROE_2014'], 
    'ROE_D_2016_2015':  df['ROE_2016'] - df['ROE_2015'], 
    'ROE_D_2017_2016':  df['ROE_2017'] - df['ROE_2016'], 
    
    'ROA_D_2013_2012':  df['ROA_2013'] - df['ROA_2012'], 
    'ROA_D_2014_2013':  df['ROA_2014'] - df['ROA_2013'], 
    'ROA_D_2015_2014':  df['ROA_2015'] - df['ROA_2014'], 
    'ROA_D_2016_2015':  df['ROA_2016'] - df['ROA_2015'], 
    'ROA_D_2017_2016':  df['ROA_2017'] - df['ROA_2016'], 
    
    'DPS_D_2013_2012':  df['DPS_2013'] - df['DPS_2012'], 
    'DPS_D_2014_2013':  df['DPS_2014'] - df['DPS_2013'], 
    'DPS_D_2015_2014':  df['DPS_2015'] - df['DPS_2014'], 
    'DPS_D_2016_2015':  df['DPS_2016'] - df['DPS_2015'], 
    'DPS_D_2017_2016':  df['DPS_2017'] - df['DPS_2016'], 
    
    'BV_D_2013_2012':  df['BV_2013'] - df['BV_2012'], 
    'BV_D_2014_2013':  df['BV_2014'] - df['BV_2013'], 
    'BV_D_2015_2014':  df['BV_2015'] - df['BV_2014'], 
    'BV_D_2016_2015':  df['BV_2016'] - df['BV_2015'], 
    'BV_D_2017_2016':  df['BV_2017'] - df['BV_2016'], 
    
    'Payout_D_2013_2012':  df['Payout_2013'] - df['Payout_2012'], 
    'Payout_D_2014_2013':  df['Payout_2014'] - df['Payout_2013'], 
    'Payout_D_2015_2014':  df['Payout_2015'] - df['Payout_2014'], 
    'Payout_D_2016_2015':  df['Payout_2016'] - df['Payout_2015'], 
    'Payout_D_2017_2016':  df['Payout_2017'] - df['Payout_2016']
})


# In[32]:


diff_df


# ## feature selection

# In[33]:


sf_d_2013_2012 = [
    'P_D_2013_2012',
    'EPS_D_2013_2012',
    'GPM_D_2013_2012',
    'Ops_D_2013_2012',
    'EBIT_D_2013_2012',
    'NPM_D_2013_2012',
    'IBD_D_2013_2012',
    'ROE_D_2013_2012',
    'ROA_D_2013_2012',
    'DPS_D_2013_2012',
    'BV_D_2013_2012',
    'Payout_D_2013_2012'
]


# In[34]:


sf_d_2014_2013 = [
    'P_D_2014_2013',
    'EPS_D_2014_2013',
    'GPM_D_2014_2013',
    'Ops_D_2014_2013',
    'EBIT_D_2014_2013',
    'NPM_D_2014_2013',
    'IBD_D_2014_2013',
    'ROE_D_2014_2013',
    'ROA_D_2014_2013',
    'DPS_D_2014_2013',
    'BV_D_2014_2013',
    'Payout_D_2014_2013'
]


# In[35]:


sf_d_2015_2014 = [
    'P_D_2015_2014',
    'EPS_D_2015_2014',
    'GPM_D_2015_2014',
    'Ops_D_2015_2014',
    'EBIT_D_2015_2014',
    'NPM_D_2015_2014',
    'IBD_D_2015_2014',
    'ROE_D_2015_2014',
    'ROA_D_2015_2014',
    'DPS_D_2015_2014',
    'BV_D_2015_2014',
    'Payout_D_2015_2014'
]


# In[36]:


sf_d_2016_2015 = [
    'P_D_2016_2015',
    'EPS_D_2016_2015',
    'GPM_D_2016_2015',
    'Ops_D_2016_2015',
    'EBIT_D_2016_2015',
    'NPM_D_2016_2015',
    'IBD_D_2016_2015',
    'ROE_D_2016_2015',
    'ROA_D_2016_2015',
    'DPS_D_2016_2015',
    'BV_D_2016_2015',
    'Payout_D_2016_2015'
]


# In[37]:


sf_d_2017_2016 = [
    'P_D_2017_2016',
    'EPS_D_2017_2016',
    'GPM_D_2017_2016',
    'Ops_D_2017_2016',
    'EBIT_D_2017_2016',
    'NPM_D_2017_2016',
    'IBD_D_2017_2016',
    'ROE_D_2017_2016',
    'ROA_D_2017_2016',
    'DPS_D_2017_2016',
    'BV_D_2017_2016',
    'Payout_D_2017_2016'
]


# ## diff corr

# In[38]:


corr_d_2013_2012 = diff_df[sf_d_2013_2012].corr()['P_D_2013_2012'].rename({
    'P_D_2013_2012' : 'P_D', 
    'EPS_D_2013_2012' : 'EPS_D', 
    'GPM_D_2013_2012' : 'GPM_D', 
    'Ops_D_2013_2012' : 'Ops_D', 
    'EBIT_D_2013_2012' : 'EBIT_D', 
    'NPM_D_2013_2012' : 'NPM_D', 
    'IBD_D_2013_2012' : 'IBD_D', 
    'ROE_D_2013_2012' : 'ROE_D', 
    'ROA_D_2013_2012' : 'ROA_D', 
    'DPS_D_2013_2012' : 'DPS_D', 
    'BV_D_2013_2012' : 'BV_D', 
    'Payout_D_2013_2012' : 'Payout_D'
})


# In[39]:


corr_d_2014_2013 = diff_df[sf_d_2014_2013].corr()['P_D_2014_2013'].rename({
    'P_D_2014_2013' : 'P_D', 
    'EPS_D_2014_2013' : 'EPS_D', 
    'GPM_D_2014_2013' : 'GPM_D', 
    'Ops_D_2014_2013' : 'Ops_D', 
    'EBIT_D_2014_2013' : 'EBIT_D', 
    'NPM_D_2014_2013' : 'NPM_D', 
    'IBD_D_2014_2013' : 'IBD_D', 
    'ROE_D_2014_2013' : 'ROE_D', 
    'ROA_D_2014_2013' : 'ROA_D', 
    'DPS_D_2014_2013' : 'DPS_D', 
    'BV_D_2014_2013' : 'BV_D', 
    'Payout_D_2014_2013' : 'Payout_D'
})


# In[40]:


corr_d_2015_2014 = diff_df[sf_d_2015_2014].corr()['P_D_2015_2014'].rename({
    'P_D_2015_2014' : 'P_D', 
    'EPS_D_2015_2014' : 'EPS_D', 
    'GPM_D_2015_2014' : 'GPM_D', 
    'Ops_D_2015_2014' : 'Ops_D', 
    'EBIT_D_2015_2014' : 'EBIT_D', 
    'NPM_D_2015_2014' : 'NPM_D', 
    'IBD_D_2015_2014' : 'IBD_D', 
    'ROE_D_2015_2014' : 'ROE_D', 
    'ROA_D_2015_2014' : 'ROA_D', 
    'DPS_D_2015_2014' : 'DPS_D', 
    'BV_D_2015_2014' : 'BV_D', 
    'Payout_D_2015_2014' : 'Payout_D'
})


# In[41]:


corr_d_2016_2015 = diff_df[sf_d_2016_2015].corr()['P_D_2016_2015'].rename({
    'P_D_2016_2015' : 'P_D', 
    'EPS_D_2016_2015' : 'EPS_D', 
    'GPM_D_2016_2015' : 'GPM_D', 
    'Ops_D_2016_2015' : 'Ops_D', 
    'EBIT_D_2016_2015' : 'EBIT_D', 
    'NPM_D_2016_2015' : 'NPM_D', 
    'IBD_D_2016_2015' : 'IBD_D', 
    'ROE_D_2016_2015' : 'ROE_D', 
    'ROA_D_2016_2015' : 'ROA_D', 
    'DPS_D_2016_2015' : 'DPS_D', 
    'BV_D_2016_2015' : 'BV_D', 
    'Payout_D_2016_2015' : 'Payout_D'
})


# In[42]:


corr_d_2017_2016 = diff_df[sf_d_2017_2016].corr()['P_D_2017_2016'].rename({
    'P_D_2017_2016' : 'P_D', 
    'EPS_D_2017_2016' : 'EPS_D', 
    'GPM_D_2017_2016' : 'GPM_D', 
    'Ops_D_2017_2016' : 'Ops_D', 
    'EBIT_D_2017_2016' : 'EBIT_D', 
    'NPM_D_2017_2016' : 'NPM_D', 
    'IBD_D_2017_2016' : 'IBD_D', 
    'ROE_D_2017_2016' : 'ROE_D', 
    'ROA_D_2017_2016' : 'ROA_D', 
    'DPS_D_2017_2016' : 'DPS_D', 
    'BV_D_2017_2016' : 'BV_D', 
    'Payout_D_2017_2016' : 'Payout_D'
})


# In[43]:


corr_d_mat = pd.DataFrame({
    'corr_d_2013_2012': corr_d_2013_2012,
    'corr_d_2014_2013': corr_d_2014_2013,
    'corr_d_2015_2014': corr_d_2015_2014,
    'corr_d_2016_2015': corr_d_2016_2015,
    'corr_d_2017_2016': corr_d_2017_2016
})


# In[44]:


corr_d_mat_sort = corr_d_mat.sort_values(by='corr_d_2015_2014', ascending=False)
corr_d_mat_sort


# In[45]:


corr_d_mat_sort.T


# ## diff ratio vs diff price

# In[46]:


plt.figure(figsize=(12,5))
plt.matshow(corr_d_mat_sort.T,fignum=1,cmap='Blues')
plt.xticks(np.arange(12), corr_d_mat_sort.T.columns,rotation=0)
plt.yticks(np.arange(5), corr_d_mat_sort.T.index,rotation=0)
plt.colorbar()


# In[47]:


sf_dnp_2013_2012 = [
    'P_2013',
    'EPS_D_2013_2012',
    'GPM_D_2013_2012',
    'Ops_D_2013_2012',
    'EBIT_D_2013_2012',
    'NPM_D_2013_2012',
    'IBD_D_2013_2012',
    'ROE_D_2013_2012',
    'ROA_D_2013_2012',
    'DPS_D_2013_2012',
    'BV_D_2013_2012',
    'Payout_D_2013_2012'
]


# In[48]:


sf_dnp_2014_2013 = [
    'P_2014',
    'EPS_D_2014_2013',
    'GPM_D_2014_2013',
    'Ops_D_2014_2013',
    'EBIT_D_2014_2013',
    'NPM_D_2014_2013',
    'IBD_D_2014_2013',
    'ROE_D_2014_2013',
    'ROA_D_2014_2013',
    'DPS_D_2014_2013',
    'BV_D_2014_2013',
    'Payout_D_2014_2013'
]


# In[49]:


sf_dnp_2015_2014 = [
    'P_2015',
    'EPS_D_2015_2014',
    'GPM_D_2015_2014',
    'Ops_D_2015_2014',
    'EBIT_D_2015_2014',
    'NPM_D_2015_2014',
    'IBD_D_2015_2014',
    'ROE_D_2015_2014',
    'ROA_D_2015_2014',
    'DPS_D_2015_2014',
    'BV_D_2015_2014',
    'Payout_D_2015_2014'
]


# In[50]:


sf_dnp_2016_2015 = [
    'P_2016',
    'EPS_D_2016_2015',
    'GPM_D_2016_2015',
    'Ops_D_2016_2015',
    'EBIT_D_2016_2015',
    'NPM_D_2016_2015',
    'IBD_D_2016_2015',
    'ROE_D_2016_2015',
    'ROA_D_2016_2015',
    'DPS_D_2016_2015',
    'BV_D_2016_2015',
    'Payout_D_2016_2015'
]


# In[51]:


sf_dnp_2017_2016 = [
    'P_2017',
    'EPS_D_2017_2016',
    'GPM_D_2017_2016',
    'Ops_D_2017_2016',
    'EBIT_D_2017_2016',
    'NPM_D_2017_2016',
    'IBD_D_2017_2016',
    'ROE_D_2017_2016',
    'ROA_D_2017_2016',
    'DPS_D_2017_2016',
    'BV_D_2017_2016',
    'Payout_D_2017_2016'
]


# In[52]:


corr_dnp_2013_2012 = diff_df[sf_dnp_2013_2012].corr()['P_2013'].rename({
    'P_2013' : 'P', 
    'EPS_D_2013_2012' : 'EPS_D', 
    'GPM_D_2013_2012' : 'GPM_D', 
    'Ops_D_2013_2012' : 'Ops_D', 
    'EBIT_D_2013_2012' : 'EBIT_D', 
    'NPM_D_2013_2012' : 'NPM_D', 
    'IBD_D_2013_2012' : 'IBD_D', 
    'ROE_D_2013_2012' : 'ROE_D', 
    'ROA_D_2013_2012' : 'ROA_D', 
    'DPS_D_2013_2012' : 'DPS_D', 
    'BV_D_2013_2012' : 'BV_D', 
    'Payout_D_2013_2012' : 'Payout_D'
})


# In[53]:


corr_dnp_2014_2013 = diff_df[sf_dnp_2014_2013].corr()['P_2014'].rename({
    'P_2014' : 'P', 
    'EPS_D_2014_2013' : 'EPS_D', 
    'GPM_D_2014_2013' : 'GPM_D', 
    'Ops_D_2014_2013' : 'Ops_D', 
    'EBIT_D_2014_2013' : 'EBIT_D', 
    'NPM_D_2014_2013' : 'NPM_D', 
    'IBD_D_2014_2013' : 'IBD_D', 
    'ROE_D_2014_2013' : 'ROE_D', 
    'ROA_D_2014_2013' : 'ROA_D', 
    'DPS_D_2014_2013' : 'DPS_D', 
    'BV_D_2014_2013' : 'BV_D', 
    'Payout_D_2014_2013' : 'Payout_D'
})


# In[54]:


corr_dnp_2015_2014 = diff_df[sf_dnp_2015_2014].corr()['P_2015'].rename({
    'P_2015' : 'P', 
    'EPS_D_2015_2014' : 'EPS_D', 
    'GPM_D_2015_2014' : 'GPM_D', 
    'Ops_D_2015_2014' : 'Ops_D', 
    'EBIT_D_2015_2014' : 'EBIT_D', 
    'NPM_D_2015_2014' : 'NPM_D', 
    'IBD_D_2015_2014' : 'IBD_D', 
    'ROE_D_2015_2014' : 'ROE_D', 
    'ROA_D_2015_2014' : 'ROA_D', 
    'DPS_D_2015_2014' : 'DPS_D', 
    'BV_D_2015_2014' : 'BV_D', 
    'Payout_D_2015_2014' : 'Payout_D'
})


# In[55]:


corr_dnp_2016_2015 = diff_df[sf_dnp_2016_2015].corr()['P_2016'].rename({
    'P_2016' : 'P', 
    'EPS_D_2016_2015' : 'EPS_D', 
    'GPM_D_2016_2015' : 'GPM_D', 
    'Ops_D_2016_2015' : 'Ops_D', 
    'EBIT_D_2016_2015' : 'EBIT_D', 
    'NPM_D_2016_2015' : 'NPM_D', 
    'IBD_D_2016_2015' : 'IBD_D', 
    'ROE_D_2016_2015' : 'ROE_D', 
    'ROA_D_2016_2015' : 'ROA_D', 
    'DPS_D_2016_2015' : 'DPS_D', 
    'BV_D_2016_2015' : 'BV_D', 
    'Payout_D_2016_2015' : 'Payout_D'
})


# In[56]:


corr_dnp_2017_2016 = diff_df[sf_dnp_2017_2016].corr()['P_2017'].rename({
    'P_2017' : 'P', 
    'EPS_D_2017_2016' : 'EPS_D', 
    'GPM_D_2017_2016' : 'GPM_D', 
    'Ops_D_2017_2016' : 'Ops_D', 
    'EBIT_D_2017_2016' : 'EBIT_D', 
    'NPM_D_2017_2016' : 'NPM_D', 
    'IBD_D_2017_2016' : 'IBD_D', 
    'ROE_D_2017_2016' : 'ROE_D', 
    'ROA_D_2017_2016' : 'ROA_D', 
    'DPS_D_2017_2016' : 'DPS_D', 
    'BV_D_2017_2016' : 'BV_D', 
    'Payout_D_2017_2016' : 'Payout_D'
})


# In[57]:


corr_dnp_mat = pd.DataFrame({
    'corr_dnp_2013': corr_dnp_2013_2012,
    'corr_dnp_2014': corr_dnp_2014_2013,
    'corr_dnp_2015': corr_dnp_2015_2014,
    'corr_dnp_2016': corr_dnp_2016_2015,
    'corr_dnp_2017': corr_dnp_2017_2016
})


# In[58]:


corr_dnp_mat_sort = corr_dnp_mat.sort_values(by='corr_dnp_2014', ascending=False)
corr_dnp_mat_sort


# In[59]:


corr_dnp_mat_sort.T


# ## diff ratio vs price

# In[60]:


plt.figure(figsize=(12,5))
plt.matshow(corr_dnp_mat_sort.T,fignum=1,cmap='Blues')
plt.xticks(np.arange(12), corr_dnp_mat_sort.T.columns,rotation=0)
plt.yticks(np.arange(5), corr_dnp_mat_sort.T.index,rotation=0)
plt.colorbar()


# # Industry group

# In[61]:


df.head()


# In[370]:


df_set = df[df['sector_code']!=13]


# In[63]:


df_mai = df[df['sector_code']==13]


# In[64]:


df_fin = df[df['sector_code']==3]


# In[65]:


df_tech = df[df['sector_code']==12]


# In[66]:


df_service_other = df[df['sector_code']==10]


# In[174]:


df_service_commerce = df[df['sector_code']==5]


# In[68]:


fig = plt.figure(1, figsize=(12, 12))
ax = fig.add_subplot()
bp = ax.boxplot([
        df_set['CAGR_P'], 
        df_mai['CAGR_P'], 
        df_fin['CAGR_P'], 
        df_tech['CAGR_P'], 
        df_service_other['CAGR_P'], 
        df_service_commerce['CAGR_P']
    ], 
    showmeans=True)


# In[69]:


df_sector = pd.read_csv('./export_value_by_sactor.csv')


# In[70]:


df_sector


# In[186]:


df_sector[df_sector['sector_code']==5]


# In[71]:


df_fin_value = df_sector[df_sector['sector_code']==3][['2012', '2013', '2014', '2015', '2016', '2017']]


# In[175]:


df_service_commerce = df_sector[df_sector['sector_code']==5][['2012', '2013', '2014', '2015', '2016', '2017']]


# In[176]:


plt.figure(figsize=(12, 12))
plt.subplot()
plt.plot(['2012', '2013', '2014', '2015', '2016', '2017'], df_fin_value.values[0].tolist(), label='fin')
plt.plot(['2012', '2013', '2014', '2015', '2016', '2017'], df_service_commerce.values[0].tolist(), label='service_commerce')
plt.suptitle('Categorical Plotting')
plt.legend()
plt.show()


# In[152]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[153]:


df_fin_value.T


# In[197]:


norm_df_fin = scaler.fit_transform(df_fin_value.T[[11]]).T


# In[198]:


norm_df_service_commerce = scaler.fit_transform(df_service_commerce.T[[32]]).T


# In[199]:


plt.figure(figsize=(12, 12))
plt.subplot()
plt.plot(['2012', '2013', '2014', '2015', '2016', '2017'], norm_df_fin.tolist()[0], label='fin')
plt.plot(['2012', '2013', '2014', '2015', '2016', '2017'], norm_df_service_commerce.tolist()[0], label='service_commerce')
plt.suptitle('Categorical Plotting')
plt.legend()
plt.show()


# # Fin sector

# In[327]:


# normalize
fin_companies = df[df['sector_code']==3][
    ['NAME', 'P_201201', 'P_201301', 'P_201401', 'P_201501', 'P_201601', 'P_201701']
]
tmp_df = fin_companies.copy().T
tmp_df.columns = a.iloc[0]
tmp_df = tmp_df.iloc[1:]
raster_arr_df = scaler.fit_transform(tmp_df)
norm_fin_companies = pd.DataFrame(data=raster_arr_df, index=tmp_df.index, columns=tmp_df.columns).T
norm_fin_companies


# In[338]:


norm_fin_companies.values[0].tolist()


# In[342]:


plt.figure(figsize=(12, 12))
plt.subplot()
for i in range(norm_fin_companies.shape[0]):
    company = norm_fin_companies.values[i].tolist()
    if company[5] > company[4] and company[4] > company[3]:
        plt.plot(['2012', '2013', '2014', '2015', '2016', '2017'], company, label=norm_fin_companies.index[i])
plt.legend()
plt.show()


# In[251]:


fin_companies


# In[741]:


plt.figure(figsize=(12, 12))
plt.subplot()
for i in range(fin_companies.shape[0]):
    company = fin_companies.values[i].tolist()
    if company[5] > company[4] and company[4] > company[3]:
        plt.plot(['2012', '2013', '2014', '2015', '2016', '2017'], company[1:], label=company[0])
        

fin_sector = df_sector[df_sector['sector_code']==3][['2012', '2013', '2014', '2015', '2016', '2017']].values[0]
plt.plot(['2012', '2013', '2014', '2015', '2016', '2017'], fin_sector/20, label='export', linestyle='--')

plt.legend()
plt.show()


# In[76]:


fin_companies


# # service_commerce sector

# In[77]:


service_commerce_companies = df[df['sector_code']==5][['NAME', 'P_201201', 'P_201301', 'P_201401', 'P_201501', 'P_201601', 'P_201701']]


# In[747]:


plt.figure(figsize=(12, 12))
plt.subplot()
for i in range(service_commerce_companies.shape[0]):
    company = service_commerce_companies.values[i].tolist()
    if company[0] != 'MAKRO' and company[6] > company[5] and company[6] > company[1]:
        plt.plot(['2012', '2013', '2014', '2015', '2016', '2017'], company[1:], label=company[0])

service_commerce_sector = df_sector[df_sector['sector_code']==5][['2012', '2013', '2014', '2015', '2016', '2017']].values[0]
plt.plot(['2012', '2013', '2014', '2015', '2016', '2017'], service_commerce_sector/200, label='export', linestyle='--')

plt.legend()
plt.show()


# In[351]:


service_commerce_companies


# In[80]:


tech_companies = df[df['sector_code']==12][['NAME', 'P_201201', 'P_201301', 'P_201401', 'P_201501', 'P_201601', 'P_201701']]


# In[364]:


plt.figure(figsize=(12, 12))
plt.subplot()
for i in range(tech_companies.shape[0]):
    company = tech_companies.values[i].tolist()
    if company[5] > company[4] and company[4] > company[3] and company[6] > company[1]:
        plt.plot(['2012', '2013', '2014', '2015', '2016', '2017'], company[1:], label=company[0])
plt.legend()
plt.show()


# In[82]:


tech_companies


# # Fin comp

# In[583]:


fin_companies[['P_201701']]


# In[587]:


fin_companies.plot.bar(x='NAME', y='P_201701', rot=0)


# In[ ]:


company


# In[593]:


plt.figure(figsize=(12, 12))
company = fin_companies.values[0].tolist()
plt.bar(['2012', '2013', '2014', '2015', '2016', '2017'], company[1:], label=company[0])
plt.legend()
plt.show()


# # KTC

# In[450]:


code = 'KTC'
ktc = pd.DataFrame({
    'NAME': ['P', 'BV', 'EPS', 'DPS'],
    '2012': df[df['NAME']==code][['P_201201', 'BV_2012', 'EPS_2012', 'DPS_2012']].values[0],
    '2013': df[df['NAME']==code][['P_201301', 'BV_2013', 'EPS_2013', 'DPS_2013']].values[0],
    '2014': df[df['NAME']==code][['P_201401', 'BV_2014', 'EPS_2014', 'DPS_2014']].values[0],
    '2015': df[df['NAME']==code][['P_201501', 'BV_2015', 'EPS_2015', 'DPS_2015']].values[0],
    '2016': df[df['NAME']==code][['P_201601', 'BV_2016', 'EPS_2016', 'DPS_2016']].values[0],
    '2017': df[df['NAME']==code][['P_201701', 'BV_2017', 'EPS_2017', 'DPS_2017']].values[0],
    '2018': [df[df['NAME']==code][['P_201801']].values[0][0], np.nan, np.nan, np.nan]
})
ktc


# In[451]:


plt.figure(figsize=(12, 12))
plt.subplot()
for i in range(ktc.shape[0]):
    ratio = ktc.values[i].tolist()
    plt.plot(['2012', '2013', '2014', '2015', '2016', '2017', '2018'], ratio[1:], label=ratio[0])
plt.legend()
plt.show()


# # GL

# In[456]:


code = 'TCAP'
stock = pd.DataFrame({
    'NAME': ['P', 'BV', 'EPS', 'DPS'],
    '2012': df[df['NAME']==code][['P_201201', 'BV_2012', 'EPS_2012', 'DPS_2012']].values[0],
    '2013': df[df['NAME']==code][['P_201301', 'BV_2013', 'EPS_2013', 'DPS_2013']].values[0],
    '2014': df[df['NAME']==code][['P_201401', 'BV_2014', 'EPS_2014', 'DPS_2014']].values[0],
    '2015': df[df['NAME']==code][['P_201501', 'BV_2015', 'EPS_2015', 'DPS_2015']].values[0],
    '2016': df[df['NAME']==code][['P_201601', 'BV_2016', 'EPS_2016', 'DPS_2016']].values[0],
    '2017': df[df['NAME']==code][['P_201701', 'BV_2017', 'EPS_2017', 'DPS_2017']].values[0],
    '2018': [df[df['NAME']==code][['P_201801']].values[0][0], np.nan, np.nan, np.nan]
})
plt.figure(figsize=(12, 12))
plt.subplot()
for i in range(stock.shape[0]):
    ratio = stock.values[i].tolist()
    plt.plot(['2012', '2013', '2014', '2015', '2016', '2017', '2018'], ratio[1:], label=ratio[0])
plt.legend()
plt.show()


# In[460]:


code = 'CPALL'
stock = pd.DataFrame({
    'NAME': ['P', 'BV', 'EPS', 'DPS'],
    '2012': df[df['NAME']==code][['P_201201', 'BV_2012', 'EPS_2012', 'DPS_2012']].values[0],
    '2013': df[df['NAME']==code][['P_201301', 'BV_2013', 'EPS_2013', 'DPS_2013']].values[0],
    '2014': df[df['NAME']==code][['P_201401', 'BV_2014', 'EPS_2014', 'DPS_2014']].values[0],
    '2015': df[df['NAME']==code][['P_201501', 'BV_2015', 'EPS_2015', 'DPS_2015']].values[0],
    '2016': df[df['NAME']==code][['P_201601', 'BV_2016', 'EPS_2016', 'DPS_2016']].values[0],
    '2017': df[df['NAME']==code][['P_201701', 'BV_2017', 'EPS_2017', 'DPS_2017']].values[0],
    '2018': [df[df['NAME']==code][['P_201801']].values[0][0], np.nan, np.nan, np.nan]
})
plt.figure(figsize=(12, 12))
plt.subplot()
for i in range(stock.shape[0]):
    ratio = stock.values[i].tolist()
    plt.plot(['2012', '2013', '2014', '2015', '2016', '2017', '2018'], ratio[1:], label=ratio[0])
plt.legend()
plt.show()


# In[604]:


code = 'TK'
stock = pd.DataFrame({
    'NAME': ['P', 'BV', 'EPS', 'DPS'],
    '2012': df[df['NAME']==code][['P_201201', 'BV_2012', 'EPS_2012', 'DPS_2012']].values[0],
    '2013': df[df['NAME']==code][['P_201301', 'BV_2013', 'EPS_2013', 'DPS_2013']].values[0],
    '2014': df[df['NAME']==code][['P_201401', 'BV_2014', 'EPS_2014', 'DPS_2014']].values[0],
    '2015': df[df['NAME']==code][['P_201501', 'BV_2015', 'EPS_2015', 'DPS_2015']].values[0],
    '2016': df[df['NAME']==code][['P_201601', 'BV_2016', 'EPS_2016', 'DPS_2016']].values[0],
    '2017': df[df['NAME']==code][['P_201701', 'BV_2017', 'EPS_2017', 'DPS_2017']].values[0],
    '2018': [df[df['NAME']==code][['P_201801']].values[0][0], np.nan, np.nan, np.nan]
})
plt.figure(figsize=(12, 12))
plt.subplot()
for i in range(stock.shape[0]):
    ratio = stock.values[i].tolist()
    plt.plot(['2012', '2013', '2014', '2015', '2016', '2017', '2018'], ratio[1:], label=ratio[0])
plt.legend()
plt.show()


# In[735]:


code = 'KTC'
stock = pd.DataFrame({
    'NAME': ['P', 'BV', 'EPS', 'DPS'],
    '2012': df[df['NAME']==code][['P_201201', 'BV_2012', 'EPS_2012', 'DPS_2012']].values[0],
    '2013': df[df['NAME']==code][['P_201301', 'BV_2013', 'EPS_2013', 'DPS_2013']].values[0],
    '2014': df[df['NAME']==code][['P_201401', 'BV_2014', 'EPS_2014', 'DPS_2014']].values[0],
    '2015': df[df['NAME']==code][['P_201501', 'BV_2015', 'EPS_2015', 'DPS_2015']].values[0],
    '2016': df[df['NAME']==code][['P_201601', 'BV_2016', 'EPS_2016', 'DPS_2016']].values[0],
    '2017': df[df['NAME']==code][['P_201701', 'BV_2017', 'EPS_2017', 'DPS_2017']].values[0],
    '2018': [df[df['NAME']==code][['P_201801']].values[0][0], np.nan, np.nan, np.nan]
})
plt.figure(figsize=(12, 12))
plt.subplot()
for i in range(stock.shape[0]):
    ratio = stock.values[i].tolist()
    plt.plot(['2012', '2013', '2014', '2015', '2016', '2017', '2018'], ratio[1:], label=ratio[0])
    

fin_sector = df_sector[df_sector['sector_code']==3][['2012', '2013', '2014', '2015', '2016', '2017']].values[0]
plt.plot(['2012', '2013', '2014', '2015', '2016', '2017'], fin_sector/100, label='export')

plt.legend()
plt.show()


# In[ ]:




