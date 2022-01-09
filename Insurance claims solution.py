#!/usr/bin/env python
# coding: utf-8

# In[287]:


import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
import seaborn as sns
import datetime


# #### 1. Import claims_data.csv and cust_data.csv which is provided to you and combine the two datasets appropriately to create a 360-degree view of the data. Use the same for the subsequent questions.

# In[288]:


claims= pd.read_csv("E:\Data analysis 360\Assignments\python\Insurance\Python Foundation Case Study 3 - Insurance Claims Case Study\claims.csv")
cust= pd.read_csv("E:\Data analysis 360\Assignments\python\Insurance\Python Foundation Case Study 3 - Insurance Claims Case Study\cust_demographics.csv")


# In[289]:


claims


# In[290]:


cust


# In[291]:


Cust_claims= claims.merge(cust, left_on=['customer_id'], right_on=['CUST_ID'], how='inner')


# In[292]:


Cust_claims.info()


# #### 2. Perform a data audit for the datatypes and find out if there are any mismatch within the current datatypes of the columns and their business significance

# In[293]:


Cust_claims.head(2)


# In[294]:


numeric_columns = Cust_claims.select_dtypes(include = ['float64', 'int64'])
object_columns = Cust_claims.select_dtypes(include = ['object'])


# In[295]:


def continuous_var_summary( x ):
    
    # freq and missings
    n_total = x.shape[0]
    n_miss = x.isna().sum()
    perc_miss = n_miss * 100 / n_total
    
    # outliers - iqr
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lc_iqr = q1 - 1.5 * iqr
    uc_iqr = q3 + 1.5 * iqr
    
    return pd.Series( [ x.dtype, x.nunique(), n_total, x.count(), n_miss, perc_miss,
                       x.sum(), x.mean(), x.std(), x.var(), 
                       lc_iqr, uc_iqr, 
                       x.min(), x.quantile(0.01), x.quantile(0.05), x.quantile(0.10), 
                       x.quantile(0.25), x.quantile(0.5), x.quantile(0.75), 
                       x.quantile(0.90), x.quantile(0.95), x.quantile(0.99), x.max() ], 
                     
                    index = ['dtype', 'cardinality', 'n_tot', 'n', 'nmiss', 'perc_miss',
                             'sum', 'mean', 'std', 'var',
                        'lc_iqr', 'uc_iqr',
                        'min', 'p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max']) 


# In[296]:


def categorical_var_summary( x ):
    
    Mode = x.value_counts().sort_values(ascending = False)[0:1].reset_index()
    return pd.Series([x.count(), x.isnull().sum(), Mode.iloc[0, 0], Mode.iloc[0, 1], 
                          round(Mode.iloc[0, 1] * 100 / x.count(), 2)], 
                     
                  index = ['n', 'nmiss', 'MODE', 'FREQ', 'PERCENT'])


# In[297]:


numeric_audit=numeric_columns.apply( lambda x: continuous_var_summary(x))
numeric_audit


# there are id variables and total_policy_claims only which can be processed as per the given data, it requires data type conversions to proper format

# In[298]:


categorical_audit= object_columns.apply( lambda x: categorical_var_summary(x))


# In[299]:


categorical_audit


# the amount variables, date variables shall be changed to proper formats, contact variable has no business significance for the analysis.

# ###### Data type conversions for variables which are not of proper type

# claim_date, claim_amount($, datatype), DateOfBirth

# In[300]:


Cust_claims["DateOfBirth"] = pd.to_datetime(Cust_claims.DateOfBirth, format = "%d-%b-%y")
Cust_claims.loc[(Cust_claims.DateOfBirth.dt.year > 2020),"DateOfBirth"]=Cust_claims[Cust_claims.DateOfBirth.dt.year > 2020]["DateOfBirth"].apply(lambda x: x - pd.DateOffset(years=100))


# In[301]:


Cust_claims["claim_date"] = pd.to_datetime(Cust_claims.claim_date, format = "%m/%d/%Y")


# #### 3. Convert the column claim_amount to numeric. Use the appropriate modules/attributes to remove the $ sign.

# In[302]:


Cust_claims["claim_amount"]= Cust_claims['claim_amount'].astype(str)
Cust_claims['claim_amount'] = Cust_claims['claim_amount'].str.replace('$','')
Cust_claims['claim_amount']= np.where(Cust_claims.claim_amount== "nan","",Cust_claims.claim_amount )


# In[303]:


Cust_claims['claim_amount'] = pd.to_numeric(Cust_claims['claim_amount'])
np.dtype(Cust_claims.claim_amount)


# #### 4. Of all the injury claims, some of them have gone unreported with the police. Create an alert flag (1,0) for all such claims.

# In[304]:


Cust_claims['unreported_claims']= np.where(Cust_claims.police_report== 'Unknown',1,0)


# In[305]:


Cust_claims.head(10)


# #### 5. One customer can claim for insurance more than once and in each claim, multiple categories of claims can be involved. However, customer ID should remain unique. 

# Retain the most recent observation and delete any duplicated records in the data based on the customer ID column.
# 

# In[306]:


Cust_claims= Cust_claims.drop_duplicates(subset="customer_id", keep= "last")


# #### 6. Check for missing values and impute the missing values with an appropriate value. (mean for continuous and mode for categorical)

# In[307]:


Cust_claims.isna().sum()


# In[308]:


Cust_claims["claim_amount"]= Cust_claims['claim_amount'].replace(np.NaN,Cust_claims['claim_amount'].mean())
Cust_claims["claim_amount"].isna().sum()


# In[309]:


Cust_claims.info()


# In[310]:


Cust_claims["total_policy_claims"] = Cust_claims['total_policy_claims'].fillna(Cust_claims['total_policy_claims'].mode()[0])


# In[311]:


Cust_claims["total_policy_claims"]= Cust_claims["total_policy_claims"].astype(str)


# In[312]:


Cust_claims["total_policy_claims"].isna().sum()


# ##### 7. Calculate the age of customers in years. Based on the age, categorize the customers according to the below criteria
# Children < 18
# 
# Youth 18-30
# 
# Adult 30-60
# 
# Senior > 60

# Some of the observations in the date of birth variable are not proper as we can observe it to be 2066

# In[313]:


curr_year = pd.to_datetime('today').year
dob_year = pd.DatetimeIndex(Cust_claims['DateOfBirth']).year          #extract year from DateOfBirth
x = dob_year-100                                               # for the years which belongs to 60's
v = curr_year - x
y = curr_year - dob_year
Cust_claims['age'] = (np.where(dob_year > curr_year,v,y))
#Categorising
Cust_claims.loc[(Cust_claims.age < 18),'AgeGroup'] = 'Children'
Cust_claims.loc[(Cust_claims.age >=18) & (Cust_claims.age <30),'AgeGroup'] = 'Youth'
Cust_claims.loc[(Cust_claims.age >=30) & (Cust_claims.age <60),'AgeGroup'] = 'Adult'
Cust_claims.loc[(Cust_claims.age >=60),'AgeGroup'] = 'Senior'


# In[314]:


Cust_claims.head(10)


# In[315]:


Cust_claims.groupby(["AgeGroup"])["age"].count()


# #### 8. What is the average amount claimed by the customers from various segments?
# 

# In[316]:


Avg_amount_segment= round(Cust_claims.groupby(["Segment"])["claim_amount"].mean(),2)
Avg_amount_segment


# #### 9. What is the total claim amount based on incident cause for all the claims that have been done at least 20 days prior to 1st of October, 2018.

# In[317]:


Amount_incedent= round(Cust_claims.loc[Cust_claims.claim_date < "2018-09-10",:].groupby("incident_cause")["claim_amount"].sum().add_prefix("total_"),2)
Amount_incedent


# #### 10. How many adults from TX, DE and AK claimed insurance for driver related issues and causes?

# In[318]:


Cust_claims.columns


# In[319]:


Adults_claims_count= Cust_claims.loc[(Cust_claims.incident_cause.str.lower().str.contains("driver") & (Cust_claims.State== "TX") | (Cust_claims.State== "DE") | (Cust_claims.State== "AK")) ].groupby(["State"])["claim_amount"].count()


# In[320]:


Adults_claims_count


# #### 11. Draw a pie chart between the aggregated value of claim amount based on gender and segment. Represent the claim amount as a percentage on the pie chart.
# 

# In[321]:


Claim_gender_segment= round(Cust_claims.groupby(["Segment", "gender"])["claim_amount"].sum().reset_index(),2)
Claim_gender_segment


# In[322]:


Claim_gender_segment_pivot= Claim_gender_segment.pivot(index="Segment", columns= "gender", values= "claim_amount")
Claim_gender_segment_pivot


# In[323]:


Claim_gender_segment_pivot.T.plot(kind="pie", subplots= True, legend= False,figsize=(20,10))
plt.show()


# #### 12. Among males and females, which gender had claimed the most for any type of driver related issues? E.g. This metric can be compared using a bar chart

# In[324]:


Claim_gender_driver= Cust_claims.loc[(Cust_claims.incident_cause.str.lower().str.contains("driver"))].groupby(["gender"])[["gender"]].count().add_prefix("countOf_").reset_index()
Claim_gender_driver


# In[325]:


sns.barplot(x= "gender", y= "countOf_gender", data= Claim_gender_driver)
plt.plot()


# #### 13. Which age group had the maximum fraudulent policy claims? Visualize it on a bar chart.

# In[326]:


Cust_claims["Frauds"]= np.where(Cust_claims.fraudulent == "Yes",1,0)


# In[329]:


AgeGroup_max_frauds= Cust_claims.groupby(["AgeGroup"])["Frauds"].sum().reset_index()
AgeGroup_max_frauds


# In[332]:


sns.barplot(x= "AgeGroup", y="Frauds", data=AgeGroup_max_frauds )
plt.plot()


# #### 14. Visualize the monthly trend of the total amount that has been claimed by the customers. Ensure that on the “month” axis, the month is in a chronological order not alphabetical order.

# In[347]:


Cust_claims['claim_month'] = pd.to_datetime(Cust_claims['claim_date'])
Cust_claims['Claim_month'] = Cust_claims['claim_month'].dt.month


# In[348]:


pd.pivot_table(Cust_claims,index="Claim_month",values="claim_amount").plot(kind='bar')
plt.ylabel('Total amount spend')
plt.show()


# #### 15. What is the average claim amount for gender and age categories and suitably represent the above using a facetted bar chart, one facet that represents fraudulent claims and the other for non-fraudulent claims.

# In[415]:


Frauds_amount_gender= pd.DataFrame(Cust_claims.loc[(Cust_claims.fraudulent=="Yes")].groupby(["gender","AgeGroup"])[["claim_amount"]].mean().add_prefix("Fraud_"))
Non_Frauds_amount_gender=pd.DataFrame(Cust_claims.loc[(Cust_claims.fraudulent=="No")].groupby(["gender","AgeGroup"])[["claim_amount"]].mean().add_prefix("Non_Fraud_"))


# In[416]:


Claims_age_gender=round(pd.merge(Frauds_amount_gender,Non_Frauds_amount_gender, on=["gender","AgeGroup"]),2)


# In[417]:


Claims_age_gender


# In[420]:


Claims_age_gender.plot(kind="bar", subplots= True, legend= True,figsize=(10,10))
plt.show()


# ##### Based on the conclusions from exploratory analysis as well as suitable statistical tests, answer the below questions. 
# Please include a detailed write-up on the parameters taken into consideration, the Hypothesistesting steps, conclusion from the p-values and the business implications of the statements.

# #### 16. Is there any similarity in the amount claimed by males and females?
# 

# In[422]:


claim_male = Cust_claims['claim_amount'].loc[Cust_claims['gender']=="Male"]
claim_female = Cust_claims['claim_amount'].loc[Cust_claims['gender']=="Female"]


# In[424]:


print("The average amount claimed by males is {}".format(claim_male.mean()))

print("The average amount claimed by females is {}".format(claim_female.mean()))


# Two sample t-test:
# To conduct a valid test: (Assumptions for two sample t-test)
# 
# * Data values must be independent. Measurements for one observation do not affect measurements for any other observation.
# * Data in each group must be obtained via a random sample from the population.
# * Data in each group are normally distributed.
# * Data values are continuous.
# * The variances for the two independent groups are equal.

# In[426]:


import scipy.stats as stats
#checking the variance

eq_var = stats.ttest_ind(a= claim_male,
                b= claim_female,
                equal_var=True)    # equal variance
eq_var.statistic


# In[428]:


uneq_var = stats.ttest_ind(a= claim_male,
                b= claim_female,
                equal_var=False)    # UnEqual variance
uneq_var.statistic


# The t score of the variables is very similar thus we will consider it as equal variance

# In[431]:


t= eq_var.statistic
p= eq_var.pvalue

print(" For the above test, the t-score is {} and the p-value is {}".format(t,p))

if(p<0.05):
    print('We reject null hypothesis')
else:
    print('We fail to reject null hypothesis')


# As the significance value of t test is greater than 0.05 we can safely conclude that there is a similarity in amount claimed by males and females.

# #### 17. Is there any relationship between age category and segment?

# To find out this relationship we will use Chi Square test

# H0= No relation between category and segment;
# Ha= There is a relationship between category and segment

# In[434]:


agecat_seg = pd.crosstab(Cust_claims.AgeGroup, Cust_claims.Segment, margins = True)
agecat_seg


# In[435]:


Chi_test = stats.chi2_contingency(observed= agecat_seg)
Chi_test


# In[437]:


print("The chi square stat is {} and the p value is {}".format(Chi_test[0],Chi_test[1]))


# Since the significance value of the test is greter than 0.05, we fail reject the null hypothesis. Therefore there is no relationship between age category and segment

# #### 18. The current year has shown a significant rise in claim amounts as compared to 2016-17 fiscal average which was $10,000.

# Here we will check the pearson coeffecient.

# The H0=  No relationship between the 2016-17 claim amounts and current claim amounts,
# Ha= Retionship exists;
# the CI= 95%, p=0.05

# In[485]:


Cust_claims["Year"]= Cust_claims['claim_month'].dt.year


# In[481]:


#current year as per the data given in 2018


# In[522]:


Current_year= Cust_claims.loc[Cust_claims.Year == 2018]["claim_amount"]
amt_2016_17= Cust_claims.loc[Cust_claims.Year==2017]["claim_amount"]


# In[527]:


#performing pearson coeffecient

stats.pearsonr(Current_year,amt_2016_17)


# In[528]:


#not able to perform it


# #### 19. Is there any difference between age groups and insurance claims?

# Here we will perform Ftest ANOVA

# H0 : mean(AgeGroup[Youth]) == mean(AgeGroup[Adult]) (No difference between age groups and insurance claims or No influence of age groups on insurance claims) Ha : mean(AgeGroup[Youth]) != mean(AgeGroup[Adult]) (There is some difference between age groups and insurance claims or there is some influence of age groups on insurance claims)

# In[531]:


age_group_1 = Cust_claims['total_policy_claims'].loc[Cust_claims['AgeGroup']=="Youth"]
age_group_2 = Cust_claims['total_policy_claims'].loc[Cust_claims['AgeGroup']=="Adult"]
# Perfrom the Anova
anova = stats.f_oneway(age_group_1,age_group_2)
# Statistic :  F Value
f = anova.statistic
p = anova.pvalue
print("The f-value is {} and the p value is {}".format(f,p))
if(p<0.05):
    print('We reject null hypothesis')
else:
    print('We fail to reject null hypothesis')


# Since the significance value of the test is greater than 0.05, we fail reject the null hypothesis. Therefore, there is no difference between age groups and insurance claims or No influence of age groups on insurance claims

# #### 20. Is there any relationship between total number of policy claims and the claimed amount?

# In[538]:


Cust_claims['total_policy_claims'] = pd.to_numeric(Cust_claims['total_policy_claims'])


# In[540]:


#Correlation
Cust_claims.total_policy_claims.corr(other= Cust_claims.claim_amount)


# As the correlation is negative the number of policy claims in inversely propotional to the claimed amount.
