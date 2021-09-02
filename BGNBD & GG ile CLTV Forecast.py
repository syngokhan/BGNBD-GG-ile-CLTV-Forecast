#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import datetime
from lifetimes import GammaGammaFitter,BetaGeoFitter
from lifetimes.plotting import plot_period_transactions

from sqlalchemy import create_engine


# In[3]:


pd.set_option("display.max_columns" , None)
pd.set_option("display.float_format" , lambda x : "%.3f" % x)
pd.set_option("display.width",200)


# In[4]:


def outlier_thresholds(dataframe,col_name, q1 = 0.01, q3 = 0.99):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile = quantile3 - quantile1
    up_limit = quantile3 + 1.5*interquantile
    low_limit = quantile1 - 1.5*interquantile
    return up_limit,low_limit


# In[5]:


def replace_with_threshols(dataframe, col_name ,q1=0.01, q3 = 0.99):
    up_limit, low_limit = outlier_thresholds(dataframe, col_name, q1 , q3)
    dataframe.loc[ (dataframe[col_name] > up_limit)  , col_name] = up_limit
    dataframe.loc[ (dataframe[col_name] < low_limit),  col_name] = low_limit


# In[6]:


path= "/Users/gokhanersoz/Desktop/VBO_Dataset/online_retail_II.xlsx"


# In[7]:


online_retail = pd.read_excel(path, sheet_name = "Year 2010-2011")


# In[8]:


df = online_retail.copy()
df.head()


# In[9]:


###########################
# EXTRA : Reading Data from Database
###########################
#host    : db.github.rocks
#port    : 3306
#user    : synan_dsmlbc_group_3_admin
#pass    : iamthedatascientist*****!
#database: synan_dsmlbc_group_3


# In[10]:


creds = {'user': 'synan_dsmlbc_group_3_admin',
         'passwd': 'iamthedatascientist*****!',
         'host': 'db.github.rocks',
         'port': 3306,
         'db': 'synan_dsmlbc_group_3'}


# In[11]:


# MySQL connection string
# pip install mysql-connector-python-rf

connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'
conn = create_engine(connstr.format(**creds))


# In[12]:


pd.read_sql_query("show databases",conn)


# In[13]:


pd.read_sql_query("show tables", conn).head()


# In[14]:


retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)
type(retail_mysql_df)


# In[15]:


retail_mysql_df.dtypes


# In[17]:


retail_mysql_df.head()


# In[18]:


retail_mysql_df.iloc[:,1:-1].head()


# In[19]:


# Let's get to our main topic.

# We examine with a describe outlier observations.
# With Quantity, Price variable has "-" values, we need to overcome them...


# In[20]:


print("DataFrame Shape : {}".format(df.shape))


# In[21]:


df.dtypes


# In[22]:


df.head()


# In[23]:


df.describe([0.01,0.99]).T


# In[24]:


# Let's see how many unique countries there are...

country = df["Country"].value_counts()
country = pd.DataFrame(country)
country.columns = ["Country Values"]
country.head()


# In[25]:


# I only shot United Kingdom since we will make 6-month CLTV prediction for UK customers....

df_uk = df[df["Country"] == "United Kingdom"]
print("DataFrame Shape : {}".format(df_uk.shape))


# In[26]:


na_values = df_uk.isnull().sum()
na_values = na_values[na_values > 0]
na_values = pd.DataFrame(na_values, columns = ["NA_Values"])
na_values


# In[27]:


# Here we see outliers for the variables Quantity and Price....
# Before using replace_with_thresholds with outlier_thresholds, I make a description and review
# Since the Min values of the Quantity and Price variables are "-" here, we need to get rid of them...

df_uk.describe([.01,.99]).T


# In[28]:


df_uk.dropna(axis = 0 , inplace = True)


# In[29]:


df_uk = df_uk[~df_uk["Invoice"].str.contains("C", na = False)]
df_uk = df_uk[df_uk["Quantity"] > 0 ]
df_uk.describe([.01, .99]).T


# In[30]:


num_cols = [col for col in df_uk.columns if df_uk[col].dtype != "object"]
for col in ["InvoiceDate","Customer ID"]:
    num_cols.remove(col)
num_cols


# In[31]:


def box_plot(dataframe, num_cols):
    
    plt.figure(figsize = (10,10))
    num = len(num_cols)
    i=1
    size = 15
    
    for col in num_cols:
        plt.subplot(num , 1, i)
        plt.boxplot(dataframe[col])
        plt.xlabel(col , fontsize = size)
        plt.ylabel("Values" , fontsize = size)
        plt.title("Outliers" , fontsize = size)
        i+=1

    plt.tight_layout()
    plt.show()



box_plot(df_uk, num_cols);


# In[32]:


df_uk.describe([.01,.99]).T


# In[33]:


# Outlier values disappear here.
# Quantity max = 80995 now max = 248.5
# Price max = 8142,750 while now max = 31.56

replace_with_threshols(df_uk ,"Quantity")
replace_with_threshols(df_uk ,"Price")
df_uk.describe([.01, .99]).T


# In[34]:


df_uk.InvoiceDate.max()


# In[35]:


# We put two days on it...
today_date = datetime.datetime(2011,12,11) 
today_date


# In[36]:


df_uk["TotalPrice"] = df_uk["Quantity"] * df_uk["Price"]
df_uk.head()


# In[37]:


df_uk.Country.unique()


# In[38]:


###########################
# Preparation of Lifetime Data Structure
###########################

## recency: The elapsed time since the last purchase. Weekly. 
##(according to analysis day on cltv_df, user specific here)

# T: The age of the customer. Weekly. (how long before the analysis date the first purchase was made)
# frequency: total number of repeat purchases (frequency>1)
# monetary_value: average earnings per purchase


# In[39]:


cltv_df = df_uk.groupby("Customer ID").agg({"InvoiceDate" : 
                                  [lambda InvoiceDate : (InvoiceDate.max()-InvoiceDate.min()).days,
                                   lambda InvoiceDate : (today_date-InvoiceDate.min()).days],
                                 
                                 "Invoice" : lambda Invoice : Invoice.nunique(),
                                  "TotalPrice" : lambda TotalPrice : TotalPrice.sum()
                                 
                                 })


# In[40]:


cltv_df.head()


# In[41]:


# # Here, let's download the first columns and fix the naming...
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns


# In[42]:


# For Rfm print("Recency : ", (today_date - test.InvoiceDate.max()).days)

values = 12747.000

test = df_uk[df_uk["Customer ID"] == values]
print(f"Values : {values}")

print("Recency : ",(test.InvoiceDate.max() - test.InvoiceDate.min()).days)
print("T : ", (today_date - test.InvoiceDate.min()).days)

print("Frequence : " , (test.Invoice.nunique()))
print("Monetary : " , (test.TotalPrice.sum()))


# In[43]:


cltv_df.columns = ["Recency" , "T", "Frequence", "Monetary"]
cltv_df.head()


# In[44]:


# Here we need to update the monetary value as the average earnings per transaction
cltv_df["Monetary"] = cltv_df["Monetary"] / cltv_df["Frequence"]
cltv_df.head()


# In[45]:


cltv_df.describe([.01, .99]).T


# In[46]:


# We need to get Monetary greater than 0.
cltv_df = cltv_df[cltv_df["Monetary"] > 0]
cltv_df.describe([.01 , .99]).T


# In[47]:


# For BGNBD, recency and T need to be expressed in weekly terms...
# According to the days, we bought it up ...days

cltv_df["Recency"] = cltv_df["Recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7


# In[48]:


#frequency must be greater than 1.
# We need to focus on at least freq greater than 1....
#1 and minors have no relationship with us anyway...

cltv_df = cltv_df[cltv_df["Frequence"] > 1]
cltv_df.describe([.01, .99]).T


# ### Mission 1:
# 
# #### 6 months CLTV Prediction
# 
# * Make a 6-month CLTV prediction for 2010-2011 UK customers.
# 
# * Interpret and evaluate the results you have obtained.
# 
# * CAUTION!
# * It is expected that cltv prediction will be made, not the expected number of transaction for 6 months.
# * So, continue by installing the BGNBD & GAMMA GAMMA models directly and enter 6 in the moon section for cltv prediction.

# In[49]:


############################################################
# Establishment of BG-NBD Model
############################################################


# In[50]:


# We applied penalty... We did it to prevent overfitting....

"""
t: array_like
     times to calculate the expectation for.
frequency: array_like
     historical frequency of customers.
recency: array_like
     historical recency of customers.
T: array_like
     age of the customer.
"""

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(frequency=cltv_df["Frequence"] , recency= cltv_df["Recency"], T = cltv_df["T"])


# In[51]:


# 1 Weekly Expected Transaction

cltv_df["Excepted_Purc_1_Week"] = bgf.predict(t = 1,
                                              frequency=cltv_df["Frequence"],
                                              recency = cltv_df["Recency"] , 
                                              T = cltv_df["T"])


# In[52]:


# 1 Month Expected Transaction

cltv_df["Excepted_Purc_1_Month"] = bgf.predict(t = 4,
                                               frequency=cltv_df["Frequence"],
                                               recency=cltv_df["Recency"],
                                               T = cltv_df["T"])


# In[53]:


# 4 Months Expected Transaction

cltv_df["Excepted_Purc_4_Month"] = bgf.predict(t = 4*4,
                                               frequency=cltv_df["Frequence"],
                                               recency=cltv_df["Recency"],
                                               T = cltv_df["T"])


# In[54]:


# 52 / 4 = 13 Months

cltv_df.head()


# In[55]:


cltv_df.sort_values(by = "Excepted_Purc_4_Month" , ascending=False).head()


# In[56]:


##############################################################
# Evaluation of Forecast Results
##############################################################

plt.figure(figsize = (15,5))
plot_period_transactions(bgf)
plt.show()


# In[57]:


############################################################
# Establishing the GAMMA-GAMMA Model
############################################################

# We find the estimated average profit per transaction of a client...

# Conditonal expected average profit will be calculated...

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(frequency= cltv_df["Frequence"],
        monetary_value=cltv_df["Monetary"])


# In[58]:


# Customers' estimated average expected profits...

ggf.conditional_expected_average_profit(frequency=cltv_df["Frequence"],
                                        monetary_value=cltv_df["Monetary"]).sort_values(ascending=False).head(10)


# In[59]:


# By looking at the results, we see that the frequency alone is not enough for its monetary value....

cltv_df["Excepted_Average_Profit"] = ggf.conditional_expected_average_profit(frequency=cltv_df["Frequence"],
                                        monetary_value=cltv_df["Monetary"])


# In[60]:


cltv_df.sort_values("Excepted_Average_Profit", ascending = False).head(20)


# In[61]:


"""
Parameters
----------
transaction_prediction_model: model
    the model to predict future transactions, literature uses
    pareto/ndb models but we can also use a different model like beta-geo models
frequency: array_like
    the frequency vector of customers' purchases
    (denoted x in literature).
recency: the recency vector of customers' purchases
         (denoted t_x in literature).
T: array_like
    customers' age (time units since first purchase)
monetary_value: array_like
    the monetary value vector of customer's purchases
    (denoted m in literature).
time: float, optional
    the lifetime expected for the user in months. Default: 12
discount_rate: float, optional
    the monthly adjusted discount rate. Default: 0.01
freq: string, optional
    {"D", "H", "M", "W"} for day, hour, month, week. This represents what unit of time your T is measure in.
    
"""

cltv_six_months=ggf.customer_lifetime_value(transaction_prediction_model=bgf,
                            frequency = cltv_df["Frequence"],
                            recency = cltv_df["Recency"],
                            T = cltv_df["T"],
                            monetary_value = cltv_df["Monetary"],
                            time = 6,# 6 Months 
                            freq="W", # T's frequency information. Weekly
                            discount_rate = 0.01 # Discount
                            ) 


# In[62]:


cltv_six_months.head()


# In[63]:


cltv_six_months.name = "Clv_Six_Months"
cltv_six_months = cltv_six_months.reset_index()
cltv_six_months.sort_values(by = "Clv_Six_Months", ascending=False).head(30)


# In[64]:


cltv_final = cltv_df.merge(cltv_six_months, on = "Customer ID", how = "left")
cltv_final.sort_values(by = "Clv_Six_Months" , ascending = False).head(10)


# In[65]:


print("CLTV Shape : {}".format(cltv_final.shape))


# * Here, we can focus on the CustomerIDs of 14088 and 14096 in the notes that will interest us. Here, although they are new compared to others, they are close compared to the old ones they left, which means that there may be a relationship between Recency and T...

# ### Mission 2:
# 
# CLTV analysis consisting of different time periods
# 
# * Calculate 1-month and 12-month CLTV for 2010-2011 UK customers.
# 
# * Analyze the top 10 people at 1-month CLTV and the top 10 people at 12 months.
# 
# * Is there a difference? If so, why do you think it could be?
# 
# * CAUTION! There is no need to build a model from scratch. It is possible to proceed over the model created in the previous question.

# In[66]:


# 1 month CLTV analysis

cltv_one_monthly = ggf.customer_lifetime_value(transaction_prediction_model= bgf,
                                               frequency= cltv_df["Frequence"],
                                               recency= cltv_df["Recency"],
                                               T=cltv_df["T"],
                                               monetary_value= cltv_df["Monetary"],
                                               time = 1,
                                               freq = "W",
                                               discount_rate=0.01)


# In[67]:


cltv_one_monthly.head()


# In[68]:


cltv_one_monthly.name = "Clv_One_Month"
cltv_one_monthly = cltv_one_monthly.reset_index()

cltv_one_month = cltv_df.merge(cltv_one_monthly, on = "Customer ID", how = "left")
cltv_one_month.sort_values(by = "Clv_One_Month" , ascending= False).head(10)


# In[69]:


# 12 months CLTV analysis

cltv_twelve_monthly = ggf.customer_lifetime_value(transaction_prediction_model= bgf,
                                                  frequency=cltv_df["Frequence"],
                                                  recency=cltv_df["Recency"],
                                                  T=cltv_df["T"],
                                                  monetary_value=cltv_df["Monetary"],
                                                  time = 12,
                                                  freq="W",
                                                  discount_rate = 0.01)


# In[70]:


cltv_twelve_monthly.head()


# In[71]:


cltv_twelve_monthly.name = "Clv_Twelve_Months"
cltv_twelve_monthly = cltv_twelve_monthly.reset_index()

cltv_twelve_month = cltv_df.merge(cltv_twelve_monthly, on = "Customer ID", how = "left")
cltv_twelve_month.sort_values(by = "Clv_Twelve_Months", ascending = False).head(10)


# In[72]:


cltv_one_month.sort_values(by = "Clv_One_Month", ascending = False).head(10)


# * CustomerId's came same. Except CustomerID at 12 months!!! If we look at the observations in the 10th place between 12 months and 1 month, two things may have happened here, either the customer in 1 month may have become churn or the parameters are not enough for our interpretation...
# 
# 
# * When we look at the 2nd and 6th rows, rather than the low frequency values, the amount of clv they have left is quite high compared to the others... There may be a relationship between the Recency and T variables.
# 
# 
# * The 1-month forecast and the 12-month CLV values were different, but this is quite normal, since the period increases on a monthly basis, the values are a little higher at 12 months. There is no difference between the two.

# ### Mission 3:
# 
# #### Segmentation and Action Recommendations
# 
# * For 2010-2011 UK customers, divide all your customers into 4 groups (segments) according to 6-month CLTV and add the group names to the dataset.
# 
# 
# * Make short 6-month action suggestions to the management for 2 groups you will choose from among 4 groups.

# In[73]:


# Standardization of CLTV

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
cltv_final["Scaled_Clv_Six_Months"] = scaler.fit_transform(cltv_final[["Clv_Six_Months"]])


# In[74]:


cltv_final.sort_values(by = "Clv_Six_Months", ascending = False).head()


# In[75]:


############################################################
# Creation of Segments by 6 Months CLTV
############################################################

cltv_final["Segment"] = pd.qcut(cltv_final["Scaled_Clv_Six_Months"], 4 , labels = ["D","C","B","A"])

cltv_final.head()


# In[77]:


cltv_final.iloc[:,1:].groupby("Segment").agg({"mean","sum"})


# * The frequency value and monetary value of the A segment are high. It has a young audience according to segments.
# 
# * We can apply campaigns in the B segment to increase it to the A segment. They can earn us more income.
# 
# * We can separate the D and C segments as a group. As a result, although they are low in return and frequency, they may have a potential for us in the future, although they are older than others in terms of recency and T values. Campaigns can be organized within them, they can be small-scale. Keeping in touch with them always opens a door of profit for us...

# ### Mission 4:
# 
# ### Sending records to database
# 
# * Send the final table, which will consist of the following variables, to the database.
# 
# * Create the name of the table as name_surname.
# 
# * The table name must be entered in the "name" section of the relevant function.

# In[78]:


cltv_final["Customer ID"] = cltv_final["Customer ID"].astype(int)

#cltv_final.to_sql(name = "Gokhan_Ersoz", con = con, if_exists="replace", index = False)

pd.read_sql_query("show tables", conn)


# In[ ]:




