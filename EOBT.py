# Importing the merge file
import pandas as pd
import numpy as np
import matplotlib


merge = pd.read_csv('E:\Study\AA\EOBT\EOBT Data/merge.csv')
GZ_BANK = pd.read_csv('E:\Study\AA\EOBT\EOBT Data/GZ_BANK.txt')# contains details of Bank

# Splittign Real SCHD Time into Date and Time in order to input bank details

merge['REAL_SCHD_DEP_TMS'] = pd.to_datetime(merge['REAL_SCHD_DEP_TMS'], errors='coerce')
# Extracting date, hour, min from datetime field
merge['Date'] = merge['REAL_SCHD_DEP_TMS'].dt.date
merge['Time_hour'] = merge['REAL_SCHD_DEP_TMS'].dt.hour
merge['Time_min'] = merge['REAL_SCHD_DEP_TMS'].dt.minute
merge['Date'] = pd.to_datetime(merge['Date'], errors='coerce')

#Converting time to numerical format to make comparion easy
merge['Time'] = merge['Time_hour']+(merge['Time_min']/100)

merge['Dep Bank'] = 100 # initiate a default value for Dep Bank helps in detecting anamoly at a later stage

# Mapping Banks data based on information received from the GZ_Bank file
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='CLT')&(merge['Time']>=16.51)&(merge['Time']<=18.59),['Dep Bank']] = 7
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='CLT')&(merge['Time']>=13.46)&(merge['Time']<=15.29),['Dep Bank']] = 5
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='CLT')&(merge['Time']>=9.01)&(merge['Time']<=10.40),['Dep Bank']] = 2
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='CLT')&(merge['Time']>=20.51)&(merge['Time']<=22.50),['Dep Bank']] = 9
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='CLT')&(merge['Time']>=12.16)&(merge['Time']<=13.45),['Dep Bank']] = 4
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='CLT')&(merge['Time']>=0.46)&(merge['Time']<=9.00),['Dep Bank']] = 1
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='CLT')&(merge['Time']>=15.30)&(merge['Time']<=16.50),['Dep Bank']] = 6
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='CLT')&(merge['Time']>=0.01)&(merge['Time']<=0.45),['Dep Bank']] = 0
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='CLT')&(merge['Time']>=10.41)&(merge['Time']<=12.15),['Dep Bank']] = 3
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='CLT')&(merge['Time']>=22.46) | (merge['Time']<=0.00),['Dep Bank']] = 10
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='CLT')&(merge['Time']>=19.00) & (merge['Time']<=20.50),['Dep Bank']] = 8

merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']>='2016-05-03')&(merge['Time']>=5.00)&(merge['Time']<=8.34),['Dep Bank']] = 1
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']>='2016-05-03')&(merge['Time']>=8.35)&(merge['Time']<=10.14),['Dep Bank']] = 2
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']>='2016-05-03')&(merge['Time']>=10.15)&(merge['Time']<=11.59),['Dep Bank']] = 3
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']>='2016-05-03')&(merge['Time']>=12.00)&(merge['Time']<=14.19),['Dep Bank']] = 4
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']>='2016-05-03')&(merge['Time']>=14.20)&(merge['Time']<=16.19),['Dep Bank']] = 5
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']>='2016-05-03')&(merge['Time']>=16.20)&(merge['Time']<=18.19),['Dep Bank']] = 6
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']>='2016-05-03')&(merge['Time']>=18.20)&(merge['Time']<=20.09),['Dep Bank']] = 7
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']>='2016-05-03')&(merge['Time']>=20.10)&(merge['Time']<=21.59),['Dep Bank']] = 8
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']>='2016-05-03')&(merge['Time']>=22.00)|(merge['Time']<=4.59),['Dep Bank']] = 9

merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']<'2016-05-03')&(merge['Time']>=1.01)&(merge['Time']<=8.44),['Dep Bank']] = 1
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']<'2016-05-03')&(merge['Time']>=8.45)&(merge['Time']<=10.09),['Dep Bank']] = 2
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']<'2016-05-03')&(merge['Time']>=10.10)&(merge['Time']<=11.44),['Dep Bank']] = 3
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']<'2016-05-03')&(merge['Time']>=11.45)&(merge['Time']<=13.29),['Dep Bank']] = 4
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']<'2016-05-03')&(merge['Time']>=13.30)&(merge['Time']<=14.59),['Dep Bank']] = 5
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']<'2016-05-03')&(merge['Time']>=15.00)&(merge['Time']<=16.54),['Dep Bank']] = 6
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']<'2016-05-03')&(merge['Time']>=16.55)&(merge['Time']<=18.14),['Dep Bank']] = 7
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']<'2016-05-03')&(merge['Time']>=18.14)&(merge['Time']<=20.04),['Dep Bank']] = 8
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']<'2016-05-03')&(merge['Time']>=20.05)&(merge['Time']<=21.39),['Dep Bank']] = 9
merge.loc[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD']=='DFW')&(merge['Date']<'2016-05-03')&(merge['Time']>=21.40)|(merge['Time']<=1.00),['Dep Bank']] = 10

#merge[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD'] == 'DFW')&(merge['Dep Bank']==100)]
#merge[(merge['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD'] == 'DFW')&(merge['Date']<'2016-05-03')].count()


merge['Dep Bank'].value_counts()
merge[merge['Dep Bank'] == 100]
merge = merge.drop(labels = ['Date','Time','Time_hour','Time_min'], axis  = 1)

# remove unamed column
merge = merge.drop(labels = ['Unnamed: 0'], axis = 1)

# Creatge new variable called NOGT
merge['NOGT'] = merge['MOGT'] - merge['AVAIL_GROUND_TM_MINUTE']
merge['MOGT>AOGT'] = 0
merge.loc[merge['NOGT']>0,'MOGT>AOGT'] = 1


# Weekend Flag
merge['weekend_flag'] =merge['day_of_week']
merge['weekend_flag'] = merge['weekend_flag'].map({1:1,7:1,2:0,3:0,4:0,5:0,6:0})

merge['weekend_flag'].value_counts()
# Assign Fleet type categories
merge['FLEET'].value_counts()

merge['fleet_type'] =merge['FLEET'] 
merge['fleet_type'] = merge['fleet_type'].map({'737':'Narrow_body','320':'Narrow_body','S80':'Narrow_body',
     '777':'Wide_body','757':'Narrow_body','767':'Wide_body','787':'Wide_body','330':'Wide_body',
     '76T':'Narrow_body','190':'Large_regional'})
merge['FLEET'].head(5)
merge['fleet_type'].value_counts()
merge['fleet_type'] = merge['fleet_type'].map({'Narrow_body':1,'Wide_body':2,'Large_regional':3})

# converting all dattime fields to correct datatype 
merge['OB_SCHD_LEG_DEP_LCL_TMS'] = pd.to_datetime(merge['OB_SCHD_LEG_DEP_LCL_TMS'], errors='coerce')
merge['OB_ACTL_LEG_DEP_LCL_TMS'] = pd.to_datetime(merge['OB_ACTL_LEG_DEP_LCL_TMS'], errors='coerce')
merge['IB_ACTL_LEG_ARVL_LCL_TMS'] = pd.to_datetime(merge['IB_ACTL_LEG_ARVL_LCL_TMS'], errors='coerce') 
merge['REAL_SCHD_DEP_TMS'] = pd.to_datetime(merge['REAL_SCHD_DEP_TMS'], errors='coerce') 
merge['ARVL'] = pd.to_datetime(merge['ARVL'], errors='coerce') 
merge['PCT_000'] = pd.to_datetime(merge['PCT_000'], errors='coerce') 
merge['PCT_005'] = pd.to_datetime(merge['PCT_005'], errors='coerce') 
merge['PCT_010'] = pd.to_datetime(merge['PCT_010'], errors='coerce') 
merge['PCT_025'] = pd.to_datetime(merge['PCT_025'], errors='coerce') 
merge['PCT_050'] = pd.to_datetime(merge['PCT_050'], errors='coerce') 
merge['PCT_075'] = pd.to_datetime(merge['PCT_075'], errors='coerce') 
merge['PCT_090'] = pd.to_datetime(merge['PCT_090'], errors='coerce') 
merge['PCT_095'] = pd.to_datetime(merge['PCT_095'], errors='coerce') 
merge['PCT_100'] = pd.to_datetime(merge['PCT_100'], errors='coerce') 


# calculatte SCHLD time to dep for all the PCT_* with respect to OB SCHD TIME 
merge['ARVL_schd_tdelta'] = (merge['ARVL'] - merge['REAL_SCHD_DEP_TMS'])*-1
merge['ARVL_schd_tdelta'] = merge['ARVL_schd_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_000_schd_tdelta'] = (merge['PCT_000'] - merge['REAL_SCHD_DEP_TMS'])*-1
merge['PCT_000_schd_tdelta'] = merge['PCT_000_schd_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_005_schd_tdelta'] = (merge['PCT_005'] - merge['REAL_SCHD_DEP_TMS'])*-1
merge['PCT_005_schd_tdelta'] = merge['PCT_005_schd_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_010_schd_tdelta'] = (merge['PCT_010'] - merge['REAL_SCHD_DEP_TMS'])*-1
merge['PCT_010_schd_tdelta'] = merge['PCT_010_schd_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_025_schd_tdelta'] = (merge['PCT_025'] - merge['REAL_SCHD_DEP_TMS'])*-1
merge['PCT_025_schd_tdelta'] = merge['PCT_025_schd_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes
 
merge['PCT_050_schd_tdelta'] = (merge['PCT_050'] - merge['REAL_SCHD_DEP_TMS'])*-1
merge['PCT_050_schd_tdelta'] = merge['PCT_050_schd_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_075_schd_tdelta'] = (merge['PCT_075'] - merge['REAL_SCHD_DEP_TMS'])*-1
merge['PCT_075_schd_tdelta'] = merge['PCT_075_schd_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_090_schd_tdelta'] = (merge['PCT_090'] - merge['REAL_SCHD_DEP_TMS'])*-1
merge['PCT_090_schd_tdelta'] = merge['PCT_090_schd_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_095_schd_tdelta'] = (merge['PCT_095'] - merge['REAL_SCHD_DEP_TMS'])*-1
merge['PCT_095_schd_tdelta'] = merge['PCT_095_schd_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_100_schd_tdelta'] = (merge['PCT_100'] - merge['REAL_SCHD_DEP_TMS'])*-1
merge['PCT_100_schd_tdelta'] = merge['PCT_100_schd_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

# Caluculate EOBT for passenger board times

merge['ARVL_tdelta'] = (merge['OB_ACTL_LEG_DEP_LCL_TMS'] - merge['ARVL'])*-1
merge['ARVL_tdelta'] = merge['ARVL_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_000_tdelta'] = (merge['PCT_000'] - merge['OB_ACTL_LEG_DEP_LCL_TMS'])*-1   
merge['PCT_000_tdelta'] = merge['PCT_000_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_005_tdelta'] = (merge['PCT_005'] - merge['OB_ACTL_LEG_DEP_LCL_TMS'])*-1 
merge['PCT_005_tdelta'] = merge['PCT_005_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_010_tdelta'] = (merge['PCT_010'] - merge['OB_ACTL_LEG_DEP_LCL_TMS'])*-1 
merge['PCT_010_tdelta'] = merge['PCT_010_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_025_tdelta'] = (merge['PCT_025'] - merge['OB_ACTL_LEG_DEP_LCL_TMS'])*-1 
merge['PCT_025_tdelta'] = merge['PCT_025_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes
 
merge['PCT_050_tdelta'] = (merge['PCT_050'] - merge['OB_ACTL_LEG_DEP_LCL_TMS'])*-1 
merge['PCT_050_tdelta'] = merge['PCT_050_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_075_tdelta'] = (merge['PCT_075'] - merge['OB_ACTL_LEG_DEP_LCL_TMS'])*-1 
merge['PCT_075_tdelta'] = merge['PCT_075_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_090_tdelta'] = (merge['PCT_090'] - merge['OB_ACTL_LEG_DEP_LCL_TMS'])*-1
merge['PCT_090_tdelta'] = merge['PCT_090_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_095_tdelta'] = (merge['PCT_095'] - merge['OB_ACTL_LEG_DEP_LCL_TMS'])*-1 
merge['PCT_095_tdelta'] = merge['PCT_095_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

merge['PCT_100_tdelta'] = (merge['PCT_100'] - merge['OB_ACTL_LEG_DEP_LCL_TMS'])*-1 
merge['PCT_100_tdelta'] = merge['PCT_100_tdelta'].astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

# Drop columns not necessary to learning phase
merge.info()
#a = merge[['IB_ACTL_LEG_ARVL_LCL_TMS','OB_SCHD_LEG_DEP_LCL_TMS','REAL_SCHD_DEP_TMS','MOGT','OB_ACTL_LEG_DEP_LCL_TMS','OB_DEP_TM_DIFF_MINUTE','Delay']]
df = merge.drop(labels = ['key','OB_SCHD_LEG_DEP_LCL_TMS','OB_ACTL_LEG_DEP_LCL_TMS','AIRCFT_REMAINED_OVER_NIGHT_IND',
                     'OB_DEP_TM_DIFF_MINUTE','IB_ACTL_LEG_ARVL_LCL_TMS','SKDOUT','UPLINE_ACT_IN',
                     'ARVL','DELAY_REASON_CD','REAL_SCHD_DEP_TMS',
                     'PCT_000','PCT_005','PCT_010','PCT_025','PCT_050',
                     'PCT_075','PCT_090','PCT_095','PCT_100'], axis = 1)


#Convert categorical columns with string value to dummy flags
df['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD'] = df['EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD'].map({'DFW':1,'CLT':0})

# Create a Flag for Delay codes
#delay_codes = list(df['DELAY_REASON_CD'].unique()) #Create list of codes
#a = {key:1 for key in delay_codes} #Create a dictionary of Key and values
#a['?'] = 0 # Replace value for ? with 0
#df['DELAY_REASON_CD'] = df['DELAY_REASON_CD'].map(a) #map the value as per the dictionary



# Create a new flag called time of the day 
#df['time_of_day'] = df['REAL_SCHD_DEP_TMS'].dt.hour
#df = df.drop(labels = ['REAL_SCHD_DEP_TMS'], axis = 1)

# Binning Time_of_day
#bins = [0,4,8,12,16,20,24]
#group_names = ['late_night','early_morning','morning','afternoon','evening','night']
#df['time_of_day_bin'] = pd.cut(df['time_of_day'],bins = bins,labels = group_names)
#df['time_of_day_bin'].value_counts()
#df = df.drop(labels = ['time_of_day'],axis =1)
#df1 = df # moving original data to new column to use for merging predictions with intact data
#df = pd.get_dummies(df,columns=['time_of_day_bin'])
#df = df.drop(labels = ['time_of_day_bin_late_night'], axis =1) # drop last bin to avoid curse of dimensionality



# converting month of year in dummy variable

#df = pd.get_dummies(df,columns=['month_of_year'])
#df = df.drop(labels = ['month_of_year_12'], axis =1)# drop last bin to avoid curse of dimensionality

df[[ 'PCT_005_tdelta', 'PCT_010_tdelta', 'PCT_025_tdelta',
    'PCT_050_tdelta', 'PCT_075_tdelta', 'PCT_090_tdelta', 'PCT_095_tdelta']].describe()


df[[ 'PCT_005_tdelta', 'PCT_010_tdelta', 'PCT_025_tdelta',
    'PCT_050_tdelta', 'PCT_075_tdelta', 'PCT_090_tdelta', 'PCT_095_tdelta']].quantile([0.01,0.05, 0.25, 0.5, 0.75,0.95, 0.99]) 
    
# Drop negative values for tdelta as passenger cannot board after flight has left and limit upper bound to 0.95 quantile
df = df[(df['PCT_005_tdelta']>=22)&(df['PCT_010_tdelta']>=21)&(df['PCT_025_tdelta']>=19)
&(df['PCT_050_tdelta']>=16)&(df['PCT_075_tdelta']>=13)&(df['PCT_090_tdelta']>=11)&(df['PCT_095_tdelta']>=9)]

df = df[(df['PCT_005_tdelta']<=55)&(df['PCT_010_tdelta']<=47)&(df['PCT_025_tdelta']<=43)
&(df['PCT_050_tdelta']<=39)&(df['PCT_075_tdelta']<=35)&(df['PCT_090_tdelta']<=31)&(df['PCT_095_tdelta']<=28)]

df = df[(df['PCT_005_tdelta']>0)&(df['PCT_010_tdelta']>0)&(df['PCT_025_tdelta']>0)
&(df['PCT_050_tdelta']>0)&(df['PCT_075_tdelta']>0)&(df['PCT_090_tdelta']>0)&(df['PCT_095_tdelta']>0)]



# Visualize the distribution of the data set

%matplotlib qt5
import seaborn as sns
#sns.pairplot(df[['OB_LENGTH_OF_HAUL','IB_LENGTH_OF_HAUL','DELAY_REASON_CD','OB_NUM_PAX','PCT_005_tdelta']])
#sns.boxplot(data = df[['OB_LENGTH_OF_HAUL','IB_LENGTH_OF_HAUL']])
#sns.boxplot(data = df[['SKDOUT','UPLINE_ACT_IN','MOGT','OB_NUM_PAX','IB_NUM_PAX']])
#sns.boxplot(data = df[['MOGT','OB_NUM_PAX','IB_NUM_PAX']])
#sns.boxplot(data = df[['OB_LENGTH_OF_HAUL','IB_LENGTH_OF_HAUL','SKDOUT','UPLINE_ACT_IN','MOGT','OB_NUM_PAX','IB_NUM_PAX']])
sns.boxplot(orient = "h",data = df[[ 'PCT_005_tdelta', 'PCT_010_tdelta', 'PCT_025_tdelta',
      'PCT_050_tdelta', 'PCT_075_tdelta', 'PCT_090_tdelta', 'PCT_095_tdelta']])


df.columns


a = df.corr()
#a.to_csv('E:\Study\AA\EOBT\EOBT Data/merge_corr.csv')




# *************PCT_005************
#Split data into training and testing
from sklearn.cross_validation import train_test_split
data = df.drop(labels = ['ARVL_tdelta','PCT_000_tdelta','PCT_005_tdelta','PCT_010_tdelta',
                         'PCT_025_tdelta','PCT_050_tdelta','PCT_075_tdelta','PCT_090_tdelta',
                         'PCT_095_tdelta','PCT_100_tdelta','MOGT','AVAIL_GROUND_TM_MINUTE','OVERNIGHT_IND'
                         ,'FLEET','ARVL_PAX','PCT_000_PAX','PCT_005_tdelta','PCT_010_PAX',
                         'PCT_025_PAX','PCT_050_PAX','PCT_075_PAX','PCT_090_PAX',
                         'PCT_095_PAX','PCT_100_PAX','NOGT','ARVL_schd_tdelta',
                         'PCT_000_schd_tdelta', 'PCT_010_schd_tdelta',
                         'PCT_025_schd_tdelta', 'PCT_050_schd_tdelta', 'PCT_075_schd_tdelta',
                         'PCT_090_schd_tdelta', 'PCT_095_schd_tdelta', 'PCT_100_schd_tdelta','IS_DEP_ARPT_INTL'], axis = 1)



a = data.corr()
labels = df['PCT_005_tdelta']
data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.3, random_state=7)

# Standardize the values for haul length
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
a = ['OB_LENGTH_OF_HAUL','IB_LENGTH_OF_HAUL','SKDOUT','UPLINE_ACT_IN','OB_NUM_PAX','IB_NUM_PAX','NOGT', 'day_of_week', 'week_of_month']

scaler.fit(data_train[a])
data_train[a] = scaler.transform(data_train[a])
data_test[a] = scaler.transform(data_test[a]) 

b = data_train.corr()
b.columns
label_train.head(5)

data_test.head(5)


# Decion Tree with automated parameter search
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import math
model = DecisionTreeRegressor()
parameter_dist = { 'max_features':["auto",'sqrt','log2',None],'max_depth': list(range(1,50)),
                  'min_samples_leaf': [100]}  
classifier = GridSearchCV(model, parameter_dist, n_jobs = -1, scoring = 'neg_mean_squared_error')
results = classifier.fit(data_train, label_train)
classifier.cv_results_
y_best_params = classifier.best_params_
y_best_score = classifier.best_score_
y = classifier.predict(data_test)
metrics.r2_score(label_test, y)
metrics.mean_absolute_error(label_test, y)
math.sqrt(metrics.mean_squared_error(label_test, y))



# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import math
model = RandomForestRegressor(random_state= 42,)
parameter_dist = {'n_estimators': [20,100,200,300], 'max_features':['auto','sqrt','log2'],
  'min_samples_leaf': [100]
}  
classifier_RFR = GridSearchCV(model, parameter_dist, n_jobs = -1, scoring = 'neg_mean_squared_error')
results = classifier_RFR.fit(data_train, label_train)
classifier_RFR.grid_scores_
y_RFR_best_params = classifier_RFR.best_params_
y_RFR_best_score = classifier_RFR.best_score_

y_RFR = classifier_RFR.predict(data_test)
metrics.r2_score(label_test, y_RFR)
metrics.mean_absolute_error(label_test, y_RFR)
math.sqrt(metrics.mean_squared_error(label_test, y_RFR))




# Neural Network Regression
from sklearn.neural_network import MLPRegressor
MLP = MLPRegressor(random_state = 42)
from scipy import stats
from sklearn.model_selection import GridSearchCV
parameter_dist={'learning_rate_init': [0.05, 0.01, 0.005, 0.001],
    'activation': ["relu", "identity", "tanh"]}

rs = GridSearchCV(MLP, parameter_dist)

rs.fit(data_train, label_train)
rs.grid_scores_
rs.best_params_
rs.scorer_

y_MLP = rs.predict(data_test)
metrics.r2_score(label_test, y_MLP)
metrics.mean_absolute_error(label_test, y_MLP)
math.sqrt(metrics.mean_squared_error(label_test, y_MLP))



# ridge regression
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=np.arange(0.1, 10, 0.1))
reg.fit(data_train, label_train)
y_ridge = reg.predict(data_test)
metrics.r2_score(label_test, y_ridge)
reg.alpha_ 
reg.coef_
metrics.r2_score(label_test, y_ridge)
metrics.mean_absolute_error(label_test, y_ridge)
math.sqrt(metrics.mean_squared_error(label_test, y_ridge))

# *************PCT_025************
#Split data into training and testing
from sklearn.cross_validation import train_test_split
data = df.drop(labels = ['ARVL_tdelta','PCT_000_tdelta','PCT_005_tdelta','PCT_010_tdelta',
                         'PCT_025_tdelta','PCT_050_tdelta','PCT_075_tdelta','PCT_090_tdelta',
                         'PCT_095_tdelta','PCT_100_tdelta','MOGT','AVAIL_GROUND_TM_MINUTE','OVERNIGHT_IND',
                         'FLEET','ARVL_PAX','PCT_000_PAX','PCT_005_PAX','PCT_010_PAX',
                         'PCT_050_PAX','PCT_075_PAX','PCT_090_PAX',
                         'PCT_095_PAX','PCT_100_PAX','NOGT','ARVL_schd_tdelta',
                         'PCT_000_schd_tdelta','PCT_005_schd_tdelta', 'PCT_010_schd_tdelta',
                         'PCT_050_schd_tdelta', 'PCT_075_schd_tdelta',
                         'PCT_090_schd_tdelta', 'PCT_095_schd_tdelta', 'PCT_100_schd_tdelta','IS_DEP_ARPT_INTL'], axis = 1)
a = data.corr()
labels = df['PCT_025_tdelta']
data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.3, random_state=7)

# Standardize the values for haul length
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
a = ['OB_LENGTH_OF_HAUL','IB_LENGTH_OF_HAUL','SKDOUT','UPLINE_ACT_IN','OB_NUM_PAX','IB_NUM_PAX','NOGT', 'day_of_week', 'week_of_month']

scaler.fit(data_train[a])
data_train[a] = scaler.transform(data_train[a])
data_test[a] = scaler.transform(data_test[a]) 

b = data_train.corr()

label_train.head(5)

# Decion Tree with automated parameter search
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
model = DecisionTreeRegressor()
parameter_dist = { 'max_features':['auto','sqrt','log2'],'max_depth': list(range(1,50)),
                  'min_samples_leaf': [100]}  
classifier_PCT_25 = GridSearchCV(model, parameter_dist, n_jobs = -1, scoring = 'neg_mean_squared_error')
classifier_PCT_25.fit(data_train, label_train)
classifier_PCT_25.best_score_
classifier_PCT_25.best_params_

y_PCT_25 = classifier_PCT_25.predict(data_test)
metrics.r2_score(label_test, y_PCT_25)
metrics.mean_absolute_error(label_test, y_PCT_25)
math.sqrt(metrics.mean_squared_error(label_test, y_PCT_25))


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import math
model = RandomForestRegressor(random_state= 42,n_jobs= -1)
parameter_dist = {'n_estimators': [20,100,200,300], 'max_features':['auto','sqrt','log2'],
  'min_samples_leaf': [100]
}  
classifier_RFR_PCT_25 = GridSearchCV(model, parameter_dist, n_jobs = -1, scoring = 'neg_mean_squared_error')
classifier_RFR_PCT_25.fit(data_train, label_train)
classifier_RFR_PCT_25.best_score_
classifier_RFR_PCT_25.best_params_
classifier_RFR_PCT_25.scorer_

y_RFR_PCT_25 = classifier_RFR_PCT_25.predict(data_test)
metrics.r2_score(label_test, y_RFR_PCT_25)
metrics.mean_absolute_error(label_test, y_RFR_PCT_25)
math.sqrt(metrics.mean_squared_error(label_test, y_RFR_PCT_25))


# Neural Network Regression
from sklearn.neural_network import MLPRegressor
MLP = MLPRegressor(random_state = 42)
from scipy import stats
from sklearn.model_selection import GridSearchCV
parameter_dist={'learning_rate_init': [0.05, 0.01, 0.005, 0.001],
    'activation': ["relu", "identity", "tanh"]}

rs_PCT_25 = GridSearchCV(MLP, parameter_dist)

rs_PCT_25.fit(data_train, label_train)
rs_PCT_25.grid_scores_
rs_PCT_25.best_params_
rs_PCT_25.scorer_

y_MLP_PCT_25 = rs_PCT_25.predict(data_test)
metrics.r2_score(label_test, y_MLP_PCT_25)
metrics.mean_absolute_error(label_test, y_MLP_PCT_25)
math.sqrt(metrics.mean_squared_error(label_test, y_MLP_PCT_25))



# ridge regression
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=np.arange(0.1, 10, 0.1))
reg.fit(data_train, label_train)
y_ridge_PCT_25 = reg.predict(data_test)
metrics.r2_score(label_test, y_ridge_PCT_25)
reg.alpha_ 
reg.coef_
metrics.r2_score(label_test, y_ridge_PCT_25)
metrics.mean_absolute_error(label_test, y_ridge_PCT_25)
math.sqrt(metrics.mean_squared_error(label_test, y_ridge_PCT_25))




# *************PCT_050************
#Split data into training and testing
from sklearn.cross_validation import train_test_split
data = df.drop(labels = ['ARVL_tdelta','PCT_000_tdelta','PCT_005_tdelta','PCT_010_tdelta',
                         'PCT_025_tdelta','PCT_050_tdelta','PCT_075_tdelta','PCT_090_tdelta',
                         'PCT_095_tdelta','PCT_100_tdelta','MOGT','AVAIL_GROUND_TM_MINUTE','OVERNIGHT_IND',
                         'FLEET','ARVL_PAX','PCT_000_PAX','PCT_005_PAX','PCT_010_PAX',
                         'PCT_025_PAX','PCT_075_PAX','PCT_090_PAX',
                         'PCT_095_PAX','PCT_100_PAX','NOGT','ARVL_schd_tdelta',
                         'PCT_000_schd_tdelta','PCT_005_schd_tdelta', 'PCT_010_schd_tdelta',
                         'PCT_025_schd_tdelta', 'PCT_075_schd_tdelta',
                         'PCT_090_schd_tdelta', 'PCT_095_schd_tdelta', 'PCT_100_schd_tdelta','IS_DEP_ARPT_INTL'], axis = 1)
a = data.corr()
labels = df['PCT_050_tdelta']
data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.3, random_state=7)

# Standardize the values for haul length
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
a = ['OB_LENGTH_OF_HAUL','IB_LENGTH_OF_HAUL','SKDOUT','UPLINE_ACT_IN','OB_NUM_PAX','IB_NUM_PAX','NOGT', 'day_of_week', 'week_of_month']

scaler.fit(data_train[a])
data_train[a] = scaler.transform(data_train[a])
data_test[a] = scaler.transform(data_test[a]) 

b = data_train.corr()

label_train.head(5)

# Decion Tree with automated parameter search
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
model = DecisionTreeRegressor()
parameter_dist = { 'max_features':['auto','sqrt','log2'],'max_depth': list(range(1,50)),
                  'min_samples_leaf': [100]}  
classifier_PCT_50 = GridSearchCV(model, parameter_dist, n_jobs = -1, scoring = 'neg_mean_squared_error')
classifier_PCT_50.fit(data_train, label_train)
classifier_PCT_50.best_score_
classifier_PCT_50.best_params_

y_PCT_50 = classifier_PCT_50.predict(data_test)
metrics.r2_score(label_test, y_PCT_50)
metrics.mean_absolute_error(label_test, y_PCT_50)
math.sqrt(metrics.mean_squared_error(label_test, y_PCT_50))


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import math
model = RandomForestRegressor(random_state= 42,n_jobs= -1)
parameter_dist = {'n_estimators': [20,100,200,300], 'max_features':['auto','sqrt','log2'],
  'min_samples_leaf': [100]
}  
classifier_RFR_PCT_50 = GridSearchCV(model, parameter_dist, n_jobs = -1, scoring = 'neg_mean_squared_error')
results = classifier_RFR_PCT_50.fit(data_train, label_train)
classifier_RFR_PCT_50.grid_scores_
classifier_RFR_PCT_50.best_params_
classifier_RFR_PCT_50.best_score_

y_RFR_PCT_50 = classifier_RFR_PCT_50.predict(data_test)
metrics.r2_score(label_test, y_RFR_PCT_50)
metrics.mean_absolute_error(label_test, y_RFR_PCT_50)
math.sqrt(metrics.mean_squared_error(label_test, y_RFR_PCT_50))



# Neural Network Regression
from sklearn.neural_network import MLPRegressor
MLP = MLPRegressor(random_state = 42)
from scipy import stats
from sklearn.model_selection import GridSearchCV
parameter_dist={'learning_rate_init': [0.05, 0.01, 0.005, 0.001],
    'activation': ["relu", "identity", "tanh"]}

rs_PCT_50 = GridSearchCV(MLP, parameter_dist)

rs_PCT_50.fit(data_train, label_train)
rs_PCT_50.grid_scores_
rs_PCT_50.best_params_
rs_PCT_50.scorer_

y_MLP_PCT_50 = rs_PCT_50.predict(data_test)
metrics.r2_score(label_test, y_MLP_PCT_50)
metrics.mean_absolute_error(label_test, y_MLP_PCT_50)
math.sqrt(metrics.mean_squared_error(label_test, y_MLP_PCT_50))



# ridge regression
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=np.arange(0.1, 10, 0.1))
reg.fit(data_train, label_train)
y_ridge_PCT_50 = reg.predict(data_test)
metrics.r2_score(label_test, y_ridge_PCT_50)
reg.alpha_ 
reg.coef_
metrics.r2_score(label_test, y_ridge_PCT_50)
metrics.mean_absolute_error(label_test, y_ridge_PCT_50)
math.sqrt(metrics.mean_squared_error(label_test, y_ridge_PCT_50))


# *************PCT_075************
#Split data into training and testing
from sklearn.cross_validation import train_test_split
data = df.drop(labels = ['ARVL_tdelta','PCT_000_tdelta','PCT_005_tdelta','PCT_010_tdelta',
                         'PCT_025_tdelta','PCT_050_tdelta','PCT_075_tdelta','PCT_090_tdelta',
                         'PCT_095_tdelta','PCT_100_tdelta','MOGT','AVAIL_GROUND_TM_MINUTE','OVERNIGHT_IND',
                         'FLEET','ARVL_PAX','PCT_000_PAX','PCT_005_PAX','PCT_010_PAX',
                         'PCT_025_PAX','PCT_050_PAX','PCT_090_PAX',
                         'PCT_095_PAX','PCT_100_PAX','NOGT','ARVL_schd_tdelta',
                         'PCT_000_schd_tdelta','PCT_005_schd_tdelta', 'PCT_010_schd_tdelta',
                         'PCT_025_schd_tdelta', 'PCT_050_schd_tdelta',
                         'PCT_090_schd_tdelta', 'PCT_095_schd_tdelta', 'PCT_100_schd_tdelta','IS_DEP_ARPT_INTL'], axis = 1)
a = data.corr()
labels = df['PCT_075_tdelta']
data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.3, random_state=7)

# Standardize the values for haul length
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
a = ['OB_LENGTH_OF_HAUL','IB_LENGTH_OF_HAUL','SKDOUT','UPLINE_ACT_IN','OB_NUM_PAX','IB_NUM_PAX','NOGT', 'day_of_week', 'week_of_month']

scaler.fit(data_train[a])
data_train[a] = scaler.transform(data_train[a])
data_test[a] = scaler.transform(data_test[a]) 

b = data_train.corr()

label_train.head(5)

# Decion Tree with automated parameter search
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
model = DecisionTreeRegressor()
parameter_dist = { 'max_features':['auto','sqrt','log2'],'max_depth': list(range(1,50)),
                  'min_samples_leaf': [100]}  
classifier_PCT_75 = GridSearchCV(model, parameter_dist, n_jobs = -1, scoring = 'neg_mean_squared_error')
classifier_PCT_75.fit(data_train, label_train)
classifier_PCT_75.best_score_
classifier_PCT_75.best_params_

y_PCT_75 = classifier_PCT_75.predict(data_test)
metrics.r2_score(label_test, y_PCT_75)
metrics.mean_absolute_error(label_test, y_PCT_75)
math.sqrt(metrics.mean_squared_error(label_test, y_PCT_75))


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import math
model = RandomForestRegressor(random_state= 42,n_jobs= -1)
parameter_dist = {'n_estimators': [20,100,200,300], 'max_features':['auto','sqrt','log2'],
  'min_samples_leaf': [100]
}  
classifier_RFR_PCT_75 = GridSearchCV(model, parameter_dist, n_jobs = -1, scoring = 'neg_mean_squared_error')
results = classifier_RFR_PCT_75.fit(data_train, label_train)
classifier_RFR_PCT_75.best_score_
classifier_RFR_PCT_75.best_params_
classifier_RFR_PCT_75.scorer_

y_RFR_PCT_75 = classifier_RFR_PCT_75.predict(data_test)
metrics.r2_score(label_test, y_RFR_PCT_75)
metrics.mean_absolute_error(label_test, y_RFR_PCT_75)
math.sqrt(metrics.mean_squared_error(label_test, y_RFR_PCT_75))


# Neural Network Regression
from sklearn.neural_network import MLPRegressor
MLP = MLPRegressor(random_state = 42)
from scipy import stats
from sklearn.model_selection import GridSearchCV
parameter_dist={'learning_rate_init': [0.05, 0.01, 0.005, 0.001],
    'activation': ["relu", "identity", "tanh"]}

rs_PCT_75 = GridSearchCV(MLP, parameter_dist)

rs_PCT_75.fit(data_train, label_train)
rs_PCT_75.grid_scores_
rs_PCT_75.best_params_
rs_PCT_75.scorer_

y_MLP_PCT_75 = rs_PCT_75.predict(data_test)
metrics.r2_score(label_test, y_MLP_PCT_75)
metrics.mean_absolute_error(label_test, y_MLP_PCT_75)
math.sqrt(metrics.mean_squared_error(label_test, y_MLP_PCT_75))



# ridge regression
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=np.arange(0.1, 10, 0.1))
reg.fit(data_train, label_train)
y_ridge_PCT_75 = reg.predict(data_test)
metrics.r2_score(label_test, y_ridge_PCT_75)
reg.alpha_ 
reg.coef_
metrics.r2_score(label_test, y_ridge_PCT_75)
metrics.mean_absolute_error(label_test, y_ridge_PCT_75)
math.sqrt(metrics.mean_squared_error(label_test, y_ridge_PCT_75))



# *************PCT_095************
#Split data into training and testing
from sklearn.cross_validation import train_test_split
data = df.drop(labels = ['ARVL_tdelta','PCT_000_tdelta','PCT_005_tdelta','PCT_010_tdelta',
                         'PCT_025_tdelta','PCT_050_tdelta','PCT_075_tdelta','PCT_090_tdelta',
                         'PCT_095_tdelta','PCT_100_tdelta','MOGT','AVAIL_GROUND_TM_MINUTE','OVERNIGHT_IND',
                         'FLEET','ARVL_PAX','PCT_000_PAX','PCT_005_PAX','PCT_010_PAX',
                         'PCT_025_PAX','PCT_050_PAX','PCT_090_PAX',
                         'PCT_075_PAX','PCT_100_PAX','NOGT','ARVL_schd_tdelta',
                         'PCT_000_schd_tdelta','PCT_005_schd_tdelta', 'PCT_010_schd_tdelta',
                         'PCT_025_schd_tdelta', 'PCT_050_schd_tdelta',
                         'PCT_090_schd_tdelta', 'PCT_075_schd_tdelta', 'PCT_100_schd_tdelta','IS_DEP_ARPT_INTL'], axis = 1)
a = data.corr()
labels = df['PCT_095_tdelta']
data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.3, random_state=7)

# Standardize the values for haul length
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
a = ['OB_LENGTH_OF_HAUL','IB_LENGTH_OF_HAUL','SKDOUT','UPLINE_ACT_IN','OB_NUM_PAX','IB_NUM_PAX','NOGT', 'day_of_week', 'week_of_month']

scaler.fit(data_train[a])
data_train[a] = scaler.transform(data_train[a])
data_test[a] = scaler.transform(data_test[a]) 

b = data_train.corr()

label_train.head(5)

# Decion Tree with automated parameter search
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
model = DecisionTreeRegressor()
parameter_dist = { 'max_features':['auto','sqrt','log2'],'max_depth': list(range(1,50)),
                  'min_samples_leaf': [100]}  
classifier_PCT_95 = GridSearchCV(model, parameter_dist, n_jobs = -1, scoring = 'neg_mean_squared_error')
classifier_PCT_95.fit(data_train, label_train)
classifier_PCT_95.best_score_
classifier_PCT_95.best_params_


y_PCT_95 = classifier_PCT_95.predict(data_test)
metrics.r2_score(label_test, y_PCT_95)
metrics.mean_absolute_error(label_test, y_PCT_95)
math.sqrt(metrics.mean_squared_error(label_test, y_PCT_95))


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import math
model = RandomForestRegressor(random_state= 42,n_jobs= -1)
parameter_dist = {'n_estimators': [20,100,200,300], 'max_features':['auto','sqrt','log2'],
  'min_samples_leaf': [100]
}  
classifier_RFR_PCT_95 = GridSearchCV(model, parameter_dist, n_jobs = -1, scoring = 'neg_mean_squared_error')
results = classifier_RFR_PCT_95.fit(data_train, label_train)
classifier_RFR_PCT_95.best_score_
classifier_RFR_PCT_95.best_params_
classifier_RFR_PCT_95.scorer_

y_RFR_PCT_95 = classifier_RFR_PCT_95.predict(data_test)
metrics.r2_score(label_test, y_RFR_PCT_95)
metrics.mean_absolute_error(label_test, y_RFR_PCT_95)
math.sqrt(metrics.mean_squared_error(label_test, y_RFR_PCT_95))


# Neural Network Regression
from sklearn.neural_network import MLPRegressor
MLP = MLPRegressor(random_state = 42)
from scipy import stats
from sklearn.model_selection import GridSearchCV
parameter_dist={'learning_rate_init': [0.05, 0.01, 0.005, 0.001],
    'activation': ["relu", "identity", "tanh"]}

rs_PCT_95 = GridSearchCV(MLP, parameter_dist)

rs_PCT_95.fit(data_train, label_train)
rs_PCT_95.grid_scores_
rs_PCT_95.best_params_
rs_PCT_95.scorer_

y_MLP_PCT_95 = rs_PCT_95.predict(data_test)
metrics.r2_score(label_test, y_MLP_PCT_95)
metrics.mean_absolute_error(label_test, y_MLP_PCT_95)
math.sqrt(metrics.mean_squared_error(label_test, y_MLP_PCT_95))



# ridge regression
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=np.arange(0.1, 10, 0.1))
reg.fit(data_train, label_train)
y_ridge_PCT_95 = reg.predict(data_test)
metrics.r2_score(label_test, y_ridge_PCT_95)
reg.alpha_ 
reg.coef_
metrics.r2_score(label_test, y_ridge_PCT_95)
metrics.mean_absolute_error(label_test, y_ridge_PCT_95)
math.sqrt(metrics.mean_squared_error(label_test, y_ridge_PCT_95))


#extracting feature importance
feature_importance = pd.DataFrame()
feature_importance['features'] = pd.Series(data_test.columns)
feature_importance.set_index(['features'])
feature_importance['PCT_005_DTR'] = pd.Series(classifier.best_estimator_.feature_importances_)
feature_importance['PCT_025_DTR'] = pd.Series(classifier_PCT_25.best_estimator_.feature_importances_)
feature_importance['PCT_050_DTR'] = pd.Series(classifier_PCT_50.best_estimator_.feature_importances_)
feature_importance['PCT_075_DTR'] = pd.Series(classifier_PCT_75.best_estimator_.feature_importances_)
feature_importance['PCT_095_DTR'] = pd.Series(classifier_PCT_95.best_estimator_.feature_importances_)
feature_importance['PCT_005_RFR'] = pd.Series(classifier_RFR.best_estimator_.feature_importances_)
feature_importance['PCT_025_RFR'] = pd.Series(classifier_RFR_PCT_25.best_estimator_.feature_importances_)
feature_importance['PCT_050_RFR'] = pd.Series(classifier_RFR_PCT_50.best_estimator_.feature_importances_)
feature_importance['PCT_075_RFR'] = pd.Series(classifier_RFR_PCT_75.best_estimator_.feature_importances_)
feature_importance['PCT_095_RFR'] = pd.Series(classifier_RFR_PCT_95.best_estimator_.feature_importances_)

feature_importance.to_excel('E:\Study\AA\EOBT\EOBT Data/feature_importance.xlsx')


# Extracting Predictions
data_test_merge = data_test.merge(pd.DataFrame(label_test), left_index = True, right_index = True, how = 'inner')

data_test_merge = data_test_merge.reset_index()

#A Merging predictions with the test dataset
data_test_merge['y'] = pd.Series(y)
data_test_merge['y_RFR'] = pd.Series(y_RFR)
data_test_merge['y_PCT_25'] = pd.Series(y_PCT_25)
data_test_merge['y_RFR_PCT_25'] = pd.Series(y_RFR_PCT_25)
data_test_merge['y_PCT_50'] = pd.Series(y_PCT_50)
data_test_merge['y_RFR_PCT_50'] = pd.Series(y_RFR_PCT_50)
data_test_merge['y_PCT_75'] = pd.Series(y_PCT_75)
data_test_merge['y_RFR_PCT_75'] = pd.Series(y_RFR_PCT_75)
data_test_merge['y_PCT_95'] = pd.Series(y_PCT_95)
data_test_merge['y_RFR_PCT_95'] = pd.Series(y_RFR_PCT_95)

data_test_merge.columns

data_test_merge =data_test_merge.drop(labels = [ 'EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD',
       'OB_LENGTH_OF_HAUL', 'IB_LENGTH_OF_HAUL', 'OB_NUM_PAX', 'IB_NUM_PAX',
       'IB_RED_EYE', 'TIGHT_TURN_IND', 'day_of_week', 'week_of_month',
       'month_of_year', 'IS_PREV_DEP_ARPT_INTL', 'IS_ARVL_ARPT_INTL',
       'IS_OB_LEG_INTL', 'IS_IB_LEG_INTL', 'PCT_095_PAX', 'Dep Bank',
       'MOGT>AOGT', 'weekend_flag', 'fleet_type', 'PCT_095_schd_tdelta',
       'PCT_095_tdelta'], axis = 1)

a = merge.merge(data_test_merge, left_index = True, right_on = 'index', how = 'inner')

a['time_of_day_bin'] = df1['time_of_day_bin']

data_test_merge.columns
a.columns


a.to_excel('E:\Study\AA\EOBT\EOBT Data/Prediction_results.xlsx')
merge.to_excel('E:\Study\AA\EOBT\EOBT Data/merge_1.xlsx')

data_train.info()