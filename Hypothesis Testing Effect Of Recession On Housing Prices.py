# libraries used -->
import pandas as pd
import numpy as np
from scipy import stats
import re

# get list of university towns -->
def get_list_of_university_towns():
    
    # Open and read files -->
    file = open('university_towns.txt',"r")
    lines = file.readlines()
    file.close()
    
    #remove empty lines -->
    new_lines = []
    for line in lines:
        if not re.match(r'^\s$',line):
            new_lines.append(line)
            
    lines = new_lines.copy()
    
    #Strip white space at beginning and end of each line -->
    for index, line in enumerate(lines):
        lines[index] = line.strip()
    
    #Loop through lines to form a dataframe -->
    data = pd.DataFrame(columns=('State','RegionName'))
    i = 0
    state_string = ""
    region_string = ""
    for line in lines:
        if '[edit]' in line:
            state_string = line.replace('[edit]',"")  
        else:
            region_string = re.sub(r' \(.*',"", line) # if it begins with (, \( extract inside replace with ""
            data.loc[i] = [state_string,region_string]
            i+=1
    return data
  
# get recession start -->
def get_recession_start():
    gdplev = pd.read_excel('gdplev.xls',skiprows=219)         
    data = gdplev.copy()
    data = data.rename(columns={data.columns[4]:'Quaters',data.columns[6]:'GDP'})
    data.set_index('Quaters',inplace=1)
    data.sort_index(axis=0,inplace=1)
    data['GDP+1'] = data.GDP.shift(-1)       #gdp of next quarter
    data['GDP-1'] = data.GDP.shift(1)        #gdp of prev quarter
    data = data[(data['GDP-1']>data['GDP']) & (data['GDP']>data['GDP+1'])]
    data = data[['GDP-1','GDP','GDP+1']].dropna()
    return data.index[0]
  
# get recession bottom -->
def get_recession_bottom():
    
    #get recession timeframe -->
    startQ = get_recession_start()
    endQ = get_recession_end()
    
    data = pd.read_excel('gdplev.xls',skiprows=219)
    data = data.rename(columns={data.columns[4]:'Quaters',data.columns[6]:'GDP'})
    data.set_index('Quaters',inplace=1)
    data.sort_index(axis=0,inplace=1)
    
    # sort in ascending order -->
    data = data[(data.index > startQ) & (data.index < endQ)]
    data.sort_values('GDP',ascending=1,inplace=1)
    return data.index[0]
  
# convert to quater -->
def convert_to_qtr(ym):
        year, month = ym.split('-')
        if month in ['01','02','03']:
            result = year + 'q1'
        elif month in ['04','05','06']:
            result = year + 'q2'
        elif month in ['07','08','09']:
            result = year + 'q3'
        else:
            result = year + 'q4'
        return result
      
# converting house data to quater -->
def convert_housing_data_to_quarters():
     
     states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama',
'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas',
'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico',
'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina',
'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 
'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}
     
     df = pd.read_csv('City_Zhvi_AllHomes.csv',header=0)
     
     # years to keep -->
     def years_to_keep(list):
             for i in range(len(list)):
                 for j in list:
                     if re.match('^19',j):
                         list.remove(j)

             return list 
             
    #Removing Unecessary Columns First -->
     cols_to_keep = list(df.columns.values)
     cols_to_keep = years_to_keep(cols_to_keep)
     cols_to_keep = [ele for ele in cols_to_keep if ele not in {'Metro','CountyName','SizeRank'}]
     df = df[cols_to_keep]
     df['State'] = df['State'].replace(states)

    # Creating a new dataframe with --> 
     df_compiled = df.copy().set_index(['State', 'RegionName', 'RegionID']).stack(dropna=False)
     df_compiled = df_compiled.reset_index().rename(columns={'level_3': 'year_month', 0: 'gdp'})
     df_compiled.drop_duplicates(inplace=True)
        
    # Converting from monthly to quaterly format -->
     df_compiled['quarter'] = df_compiled['year_month'].apply(convert_to_qtr)
     df_compiled = df_compiled.drop('year_month', axis=1)    
    
    # return result dataframe -->
     result = df_compiled.pivot_table(values='gdp', index=['State', 'RegionName', 'RegionID'], columns='quarter', aggfunc=np.mean)
     result = result.reset_index()
     result = result.drop('RegionID', axis=1)
     result = result.set_index(['State', 'RegionName'])
     return result    
  
# Get All relevant constraints -->
housing_data = convert_housing_data_to_quarters()
startQ = get_recession_start()
bottomQ = get_recession_bottom()
endQ = get_recession_end()

# Recession Timeframe -->
housing_data['Delta'] = housing_data[bottomQ] - housing_data[startQ]
  
# Tag University Towns As True -->
uni_regions = get_list_of_university_towns()
uni_regions['Uni_regions'] = True
        
# merge housing data with uni_town df --?
data2 = pd.merge(housing_data,uni_regions,how='right', left_index=True,right_on=['State', 'RegionName'])
uni_towns = data2[data2['Uni_regions']==True]['Delta']
            
# merge housing data with non-uni_town df -->
data3 = pd.merge(data,uni_regions,how='left',left_index=True,right_on=['State','RegionName'])
data3['Uni_regions'] = data3['Uni_regions'].replace({np.NaN: False})
non_uni_towns = data3[data3['Uni_regions'] == False]['Delta']

# Running ttest -->
def run_ttest():
        #Get All relevant constraints -->
        housing_data = convert_housing_data_to_quarters()
        startQ = get_recession_start()
        bottomQ = get_recession_bottom()
        endQ = get_recession_end()

        #Recession Timeframe -->
        housing_data['Delta'] = housing_data[bottomQ]- housing_data[startQ]
                
        #Regions with Universities in them -->
        uni_regions = get_list_of_university_towns()
        uni_regions['Uni_regions'] = True        
        
        #merge housing data with uni_town df -->
        data2 = pd.merge(housing_data,uni_regions,how='right', left_index=True,right_on=['State', 'RegionName'])
        uni_towns = data2[data2['Uni_regions']==True]['Delta']
            
        #merge housing data with non-uni_town df -->
        data3 = pd.merge(data,uni_regions,how='left',left_index=True,right_on=['State','RegionName'])
        data3['Uni_regions'] = data3['Uni_regions'].replace({np.NaN: False})
        non_uni_towns = data3[data3['Uni_regions'] == False]['Delta']

        #Run ttest on both dataframes -->
        st,p = stats.ttest_ind(uni_towns,non_uni_towns,nan_policy='omit')
        
        #get different p-values (bar at 0.01) -->
        different = False 
        if p < 0.01:
            different = True     # p<0.01 means lower than 1% of the thing happening by chance.
         
        #Which place is better -->
        better = ""
        if uni_towns.mean() > non_uni_towns.mean():
            better = "university town"
        else:
            better = "non-university towns"
        
        return (different,p,better)
            
            
run_ttest()