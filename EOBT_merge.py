# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:25:55 2017

@author: anubhav.reddy
"""

import pandas as pd
import numpy as np
import matplotlib

#importing data from txt files
eobt_data = pd.read_csv('E:\Study\AA\EOBT\EOBT Data/EOBT_FLT_DATA.txt', sep = '\t')
eobt_events = pd.read_csv('E:\Study\AA\EOBT\EOBT Data/EOBT_FLT_EVENTS.txt', sep = '\t')

#exploring data
eobt_data.info()
eobt_events.info()
# Converting feature to datetime format
eobt_events['OB_SCHD_LEG_DEP_LCL_DT'] = pd.to_datetime(eobt_events['OB_SCHD_LEG_DEP_LCL_DT'], errors='coerce')
eobt_events['EVENT_TMS'] = pd.to_datetime(eobt_events['EVENT_TMS'], errors='coerce')


#Splitting Events file into DFW and CLT
eobt_event_dfw1 = eobt_events[(eobt_events['OB_SCHD_LEG_DEP_AIRPRT_IATA_CD'] == 'DFW') 
                            & (eobt_events['OB_SCHD_LEG_DEP_LCL_DT'] <= '12/1/2016')]

eobt_event_dfw2 = eobt_events[(eobt_events['OB_SCHD_LEG_DEP_AIRPRT_IATA_CD'] == 'DFW') 
                            & (eobt_events['OB_SCHD_LEG_DEP_LCL_DT'] > '12/1/2016')]

eobt_event_clt = eobt_events[eobt_events['OB_SCHD_LEG_DEP_AIRPRT_IATA_CD'] == 'CLT']

#writing the file out into excel in order to connect the data from flt_data file
eobt_event_dfw1.to_excel('E:\Study\AA\EOBT\EOBT Data/eobt_event_dfw1.xlsx')
eobt_event_dfw2.to_excel('E:\Study\AA\EOBT\EOBT Data/eobt_event_dfw2.xlsx')
eobt_event_clt.to_excel('E:\Study\AA\EOBT\EOBT Data/eobt_event_clt.xlsx')


# The exported file have been put into access and merged

# Import the file generated through the access DB
import pandas as pd
import numpy as np
import matplotlib

#Loading Combined data into the Pyhton
combined_data = pd.read_csv('E:\Study\AA\EOBT\EOBT Data/Combined_Data.txt', sep = '\t')

# Create a new key by adding fleet to existing key
combined_data['key'] = combined_data['EOBT_FLT_DATA_Key'] + combined_data['FLEET']

#Select only necessary columns and explore the dataframe to understand the composiiton of each column
df = combined_data[['key',
'EOBT_FLT_DATA_OB_SCHD_LEG_DEP_AIRPRT_IATA_CD',
'OB_SCHD_LEG_DEP_LCL_TMS',
'OB_ACTL_LEG_DEP_LCL_TMS',
'OB_LENGTH_OF_HAUL',
'AIRCFT_REMAINED_OVER_NIGHT_IND',
'OB_DEP_TM_DIFF_MINUTE',
'IB_LENGTH_OF_HAUL',
'IB_ACTL_LEG_ARVL_LCL_TMS',
'MOGT','SKDOUT','UPLINE_ACT_IN','AVAIL_GROUND_TM_MINUTE','DELAY_REASON_CD','OB_NUM_PAX','IB_NUM_PAX',
'IB_RED_EYE',
'OVERNIGHT_IND',
'TIGHT_TURN_IND',
'day_of_week',
'week_of_month',
'month_of_year',
'IS_PREV_DEP_ARPT_INTL',
'IS_DEP_ARPT_INTL',
'IS_ARVL_ARPT_INTL',
'IS_OB_LEG_INTL',
'IS_IB_LEG_INTL',
'EVENT_TMS',
'EVENT_NM','FLEET','num_pax','num_pax_boarded']]

df.head(5)
df.isnull().any()
df.info()

#Assigning the correct datatype
df.loc[:,'OB_SCHD_LEG_DEP_LCL_TMS'] = pd.to_datetime(df['OB_SCHD_LEG_DEP_LCL_TMS'], errors='coerce')
#df.loc[:,'OB_ACTL_LEG_DEP_LCL_TMS'] = pd.to_datetime(df['OB_ACTL_LEG_DEP_LCL_TMS'], errors='coerce')
df.loc[:,'IB_ACTL_LEG_ARVL_LCL_TMS'] = pd.to_datetime(df['IB_ACTL_LEG_ARVL_LCL_TMS'], errors='coerce') 
#df.loc[:,'EVENT_TMS'] = pd.to_datetime(df['EVENT_TMS'], errors='coerce')

#Calculating Real Scheduled Departure time based on IB_ACTL_LEG_ARVL_LCL_TMS + MOGT
df['REAL_SCHD_DEP_TMS'] = df['IB_ACTL_LEG_ARVL_LCL_TMS'] + pd.to_timedelta(df.MOGT, unit='m')

# Assigning Real SCHD Dep time = max (Calculated dep time, OB SCHD Dep Time)
df['REAL_SCHD_DEP_TMS'] = df[['REAL_SCHD_DEP_TMS','OB_SCHD_LEG_DEP_LCL_TMS']].max(axis = 1)

#Cross Check if the max column works??
#a.head(5)
#df['REAL_SCHD_DEP_TMS'].head(5)
#df['OB_SCHD_LEG_DEP_LCL_TMS'].head(5)

# Calculate the Real Delay = Actual Departure Time - Real Scheduled Departure Time
#a = df['OB_ACTL_LEG_DEP_LCL_TMS'] - df['REAL_SCHD_DEP_TMS']
#df['Delay'] = a.astype('timedelta64[m]') #adding a column to dataframe with delay in minutes

# Cross check the values of Real Delay and Dep Time Diff
#df[['Delay','OB_DEP_TM_DIFF_MINUTE']]

# Claculating a column with timestamp 45 minutes prior to departure
#df['45_TO_SCHD_DEP_TMS'] = df['REAL_SCHD_DEP_TMS'] - pd.to_timedelta(45, unit='m')
#df['35_TO_SCHD_DEP_TMS'] = df['REAL_SCHD_DEP_TMS'] - pd.to_timedelta(35, unit='m')
#df['25_TO_SCHD_DEP_TMS'] = df['REAL_SCHD_DEP_TMS'] - pd.to_timedelta(25, unit='m')
#df['15_TO_SCHD_DEP_TMS'] = df['REAL_SCHD_DEP_TMS'] - pd.to_timedelta(15, unit='m')
#df['5_TO_SCHD_DEP_TMS'] = df['REAL_SCHD_DEP_TMS'] - pd.to_timedelta(5, unit='m')

# We need to extract the Passenger onboard percentage timstamp and reshape the data to make PCT_* as columns with
# difference betwene the board time and real SCHD time as values in the columns

# Calculate passenger remaining to board
df['pax_left'] = df['num_pax']-df['num_pax_boarded']


# Exract passenger onboard data with time stamp
pivot_PCT = df[['key','EVENT_NM','EVENT_TMS','pax_left']]



# count the number of events for each key
pivot_PCT['key'].value_counts()

# Create a new column wih key count data
pivot_PCT['Counts'] = pivot_PCT.groupby(['key'])['EVENT_NM'].transform('count')

# select only the keys with 10 counts to prevent pivot function from throwing an error
pivot_PCT = pivot_PCT[pivot_PCT['Counts']==10]

#e = e[(e['key']=='12/11/2016438DFWS80')|(e['key']=='8/31/20162311DFW737')]

# drop Counts column to rehape the events
pivot_PCT = pivot_PCT.drop(labels = 'Counts', axis =1)

# reshape the data into pivot where PCT_* is in coumns and respective time stamp as values
PAX_LEFT = pivot_PCT.pivot(index = 'key', columns = 'EVENT_NM', values = 'pax_left')
pivot_PCT = pivot_PCT.pivot(index = 'key', columns = 'EVENT_NM', values = 'EVENT_TMS')

#Extract the flight details from the Combined file

df1 = df.drop(labels = ['EVENT_TMS','EVENT_NM','pax_left','num_pax','num_pax_boarded'], axis=1)
df1 = df1.drop_duplicates()

# merge the pivot table with the flight data to get flight level pivot 
merge = df1.merge(pivot_PCT, left_on = 'key', right_index = True, how = 'inner')
merge = merge.merge(PAX_LEFT,left_on = 'key', right_index = True, how = 'inner')
merge = merge.reset_index()
merge = merge.drop(labels = 'index', axis =1)

# rename columns
merge.columns
merge = merge.rename(index=str, columns={'ARVL_x': 'ARVL', 'PCT_000_x':'PCT_000', 'PCT_005_x':'PCT_005', 'PCT_010_x':'PCT_010',
       'PCT_025_x':'PCT_025', 'PCT_050_x':'PCT_050', 'PCT_075_x':'PCT_075', 'PCT_090_x':'PCT_090', 'PCT_095_x':'PCT_095',
       'PCT_100_x':'PCT_100','ARVL_y':'ARVL_PAX', 'PCT_000_y':'PCT_000_PAX', 'PCT_005_y':'PCT_005_PAX', 'PCT_010_y':'PCT_010_PAX',
       'PCT_025_y':'PCT_025_PAX', 'PCT_050_y':'PCT_050_PAX', 'PCT_075_y':'PCT_075_PAX', 'PCT_090_y':'PCT_090_PAX', 'PCT_095_y':'PCT_095_PAX',
       'PCT_100_y':'PCT_100_PAX'})

#printout the data to text file as uncertain about the future calculations
merge.to_csv('E:\Study\AA\EOBT\EOBT Data/merge.csv')



