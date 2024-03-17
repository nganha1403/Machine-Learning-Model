import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
import sklearn

#Titles
#tit1,tit2 = st.beta_columns((4, 1))
tit1 = st.title('Machine Learning in Economy') 
#tit2.image("healthcare2.png")
st.sidebar.title("Dataset and Classifier")


dataset_name=st.sidebar.selectbox("Select data need to predict: ",('GDP',"Unemployment"))

def get_dataset():
        data = pd.read_csv('Data WorldBank fix.csv')
        return data

if dataset_name=="GDP":
    
    st.write("Kiểm tra độ tương quan giữa các cột bằng biểu đồ nhiệt")
    data = get_dataset()
    fig = plt.figure(figsize=(18,6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidth=1, linecolor='r')
    st.pyplot(fig)
    st.write('------------------------------------------------------------------------------------------')
    
    #data = pd.read_csv('Data WorldBank fix.csv')
        

    st.header("GDP Prediction")
    att_regn = st.sidebar.selectbox('Region', options=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25))
    att_infla = st.sidebar.number_input('Inflation, GDP deflator (annual %) (Example: 10)', min_value=-100.00, max_value=100.00, value=0.00, step=0.01)
    att_exch = st.sidebar.number_input('Official exchange rate (LCU per US$, period average) (Example: 10000)', min_value=1.00, max_value=100000.00, value=1.00,step=0.01)
    att_money =st.sidebar.number_input('Money supply (percent of GDP) (Example: 10)',min_value=10.00, max_value=10000000.00, value=10.00, step=0.01)
    att_unempl = st.sidebar.number_input('Unemployment, total rate of labor force) (modeled ILO estimate)',min_value=0.00, max_value=9999999999999.00, value=0.00, step=0.01)
    att_fdi = st.sidebar.number_input('Foreign direct investment, net (BoP, current US$) (Example: 10)',min_value=-40000000000.00, max_value=50000000000.00, value=-40000000000.00, step=0.01)
    
    st.write(''' Country list:  
         * 1: Brunei Darussalam
         * 2: Cambodia
         * 3: Timor-Leste
         * 4: Indonesia
         * 5: Lao PDR
         * 6: Malaysia
         * 7: Myanmar
         * 8: Philippines
         * 9: Singapore
         * 10: Thailand
         * 11: Viet Nam
         * 12: China
         * 13: Japan 
         * 14: Korea, Rep.
         * 15: India
         * 16: Uzbakistan
         * 17: Qatar 
         * 18: United States
         * 19: United Kingdom
         * 20: France
         * 21: Russian Federation
         * 22: Germany
         * 23: Australia
         * 24: Hungary
         * 25: Ireland
        
        
         '''
         )

    
    if att_regn == 1:
        att_regn_1 = 1
        att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25  = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Brunei Darussalam',["GDP"]]
        data_year = data.loc[data['Country '] == 'Brunei Darussalam',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Brunei Darussalam')
        st.pyplot(fig)
    elif att_regn == 2: 
        att_regn_2 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Timor-Leste',["GDP"]]
        data_year = data.loc[data['Country '] == 'Timor-Leste',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Timor-Leste')
        st.pyplot(fig)
    elif att_regn == 3: 
        att_regn_3 = 1
        att_regn_1 = att_regn_2 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Cambodia',["GDP"]]
        data_year = data.loc[data['Country '] == 'Cambodia',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Cambodia')
        st.pyplot(fig)
    elif att_regn == 4: 
        att_regn_4 = 1
        att_regn_1 = att_regn_2= att_regn_3 =  att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Indonesia',["GDP"]]
        data_year = data.loc[data['Country '] == 'Indonesia',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Indonesia')
        st.pyplot(fig)
    elif att_regn == 5: 
        att_regn_5 = 1
        att_regn_1 = att_regn_2= att_regn_3 = att_regn_4 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Lao PDR',["GDP"]]
        data_year = data.loc[data['Country '] == 'Lao PDR',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Lao PDR')
        st.pyplot(fig)
    elif att_regn == 6: 
        att_regn_6 = 1
        att_regn_1 = att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Malaysia',["GDP"]]
        data_year = data.loc[data['Country '] == 'Malaysia',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Malaysia')
        st.pyplot(fig)
    elif att_regn == 7: 
        att_regn_7 = 1
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Myanmar',["GDP"]]
        data_year = data.loc[data['Country '] == 'Myanmar',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Myanmar')
        st.pyplot(fig)
    elif att_regn == 8: 
        att_regn_8 = 1
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Philippines',["GDP"]]
        data_year = data.loc[data['Country '] == 'Philippines',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Philippines')
        st.pyplot(fig)
    elif att_regn == 9: 
        att_regn_9 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_2 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Singapore',["GDP"]]
        data_year = data.loc[data['Country '] == 'Singapore',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Singapore')
        st.pyplot(fig)
    elif att_regn == 10: 
        att_regn_10 = 1
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 =  att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Thailand',["GDP"]]
        data_year = data.loc[data['Country '] == 'Thailand',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Thailand')
        st.pyplot(fig)
    elif att_regn == 11: 
        att_regn_11 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Viet Nam',["GDP"]]
        data_year = data.loc[data['Country '] == 'Viet Nam',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Viet Nam')
        st.pyplot(fig)
    elif att_regn == 12: 
        att_regn_12 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'China',["GDP"]]
        data_year = data.loc[data['Country '] == 'China',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of China')
        st.pyplot(fig)
    elif att_regn == 13: 
        att_regn_13 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Japan',["GDP"]]
        data_year = data.loc[data['Country '] == 'Japan',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Japan')
        st.pyplot(fig)
    elif att_regn == 14: 
        att_regn_14 = 1    
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Korea, Rep.',["GDP"]]
        data_year = data.loc[data['Country '] == 'Korea, Rep.',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Korea, Rep.')
        st.pyplot(fig)
    elif att_regn == 15: 
        att_regn_15 = 1 
        att_regn_1 = att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'India',["GDP"]]
        data_year = data.loc[data['Country '] == 'India',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of India')
        st.pyplot(fig)
    elif att_regn == 16: 
        att_regn_16 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Uzbakistan',["GDP"]]
        data_year = data.loc[data['Country '] == 'Uzbakistan',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Uzbakistan')
        st.pyplot(fig)
    elif att_regn == 17: 
        att_regn_17 = 1 
        att_regn_1 = att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Qatar',["GDP"]]
        data_year = data.loc[data['Country '] == 'Qatar',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Qatar')
        st.pyplot(fig)
    elif att_regn == 18: 
        att_regn_18 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'United States',["GDP"]]
        data_year = data.loc[data['Country '] == 'United States',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of United States')
        st.pyplot(fig)
    elif att_regn == 19: 
        att_regn_19 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'United Kingdom',["GDP"]]
        data_year = data.loc[data['Country '] == 'United Kingdom',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of United Kingdom')
        st.pyplot(fig)
    elif att_regn == 20: 
        att_regn_20 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'France',["GDP"]]
        data_year = data.loc[data['Country '] == 'France',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of France')
        st.pyplot(fig)
    elif att_regn == 21: 
        att_regn_21 = 1    
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Russian Federation',["GDP"]]
        data_year = data.loc[data['Country '] == 'Russian Federation',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Russian Federation')
        st.pyplot(fig)
    elif att_regn == 22: 
        att_regn_22 = 1    
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Germany',["GDP"]]
        data_year = data.loc[data['Country '] == 'Germany',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Germany')
        st.pyplot(fig)
    elif att_regn == 23: 
        att_regn_23 = 1  
        att_regn_1 = att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_24 = att_regn_25 = 0   
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Australia',["GDP"]]
        data_year = data.loc[data['Country '] == 'Australia',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Australia')
        st.pyplot(fig)
    elif att_regn == 24:
        att_regn_24 = 1
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_25 = 0 
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Hungary',["GDP"]]
        data_year = data.loc[data['Country '] == 'Hungary',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Hungary')
        st.pyplot(fig)
    else:
        att_regn_25 = 1
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 =0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Ireland',["GDP"]]
        data_year = data.loc[data['Country '] == 'Ireland',["Year"]]
        df_gdp = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('GDP')
        ax.plot(df_gdp['Year'], df_gdp['GDP'])
        ax.set_title('GDP of Ireland')
        st.pyplot(fig)
    
    user_input = np.array([att_infla,att_exch,att_money,att_unempl,att_fdi,att_regn_1,att_regn_2, att_regn_3,
                       att_regn_4, att_regn_5, att_regn_6, att_regn_7, 
                       att_regn_8, att_regn_9, att_regn_10, att_regn_11, att_regn_12, att_regn_13, att_regn_14, att_regn_15,
                       att_regn_16, att_regn_17, att_regn_18, att_regn_19, att_regn_20, att_regn_21, att_regn_22, att_regn_23, att_regn_24,att_regn_25]).reshape(1,-1)
        
    if st.sidebar.button('Estimate GDP'):
        data = get_dataset()
        data['Country '] = data['Country '].astype('category')
        data_final = pd.concat([data,pd.get_dummies(data['Country '], prefix='Country ')], axis=1).drop(['Country '],axis=1)
    
        y = data_final["GDP"]
        X = data_final.drop(['Year','GDP'], axis=1)
        #X = data_final
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    
        gbm_opt = GradientBoostingRegressor(learning_rate=0.01, n_estimators=500,
                                        max_depth=5, min_samples_split=10, 
                                        min_samples_leaf=1, subsample=0.7,
                                        max_features=28, random_state=101)
        gbm_opt.fit(X_train,y_train)
        predictions = gbm_opt.predict(X_test)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=X_test, mode='markers', name='Prediction'))
        fig.update_layout(title='Scatter Plot of Y_Test vs Prediction',
                  xaxis_title='Y_Test',
                  yaxis_title='X_test')
        st.plotly_chart(fig)
    
        #making a prediction
        gbm_predictions = gbm_opt.predict(user_input)
        st.write('The estimated GDP per capita is: ', gbm_predictions)
        
else:
    st.header("Unemployment Prediction")
    st.write("Kiểm tra độ tương quan giữa các cột bằng biểu đồ nhiệt")
    data = get_dataset()
    fig = plt.figure(figsize=(18,6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidth=1, linecolor='r')
    st.pyplot(fig)
    st.write('------------------------------------------------------------------------------------------')
    att_regn = st.sidebar.selectbox('Region', options=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25))
    att_infla = st.sidebar.number_input('Inflation, GDP deflator (annual %) (Example: 10)', min_value=-100.00, max_value=100.00, value=0.00, step=0.01)
    att_exch = st.sidebar.number_input('Official exchange rate (LCU per US$, period average) (Example: 10000)', min_value=1.00, max_value=100000.00, value=1.00,step=0.01)
    att_money =st.sidebar.number_input('Money supply (percent of GDP) (Example: 10)',min_value=10.00, max_value=10000000.00, value=10.00, step=0.01)
    att_gdp = st.sidebar.number_input('GDP Growth',min_value=0.00, max_value=9999999999999.00, value=0.00, step=0.01)
    att_fdi = st.sidebar.number_input('Foreign direct investment, net (BoP, current US$) (Example: 10)',min_value=-40000000000.00, max_value=50000000000.00, value=-40000000000.00, step=0.01)
    
    st.write('''Country list:
         * 1: Brunei Darussalam
         * 2: Cambodia
         * 3: Timor-Leste
         * 4: Indonesia
         * 5: Lao PDR
         * 6: Malaysia
         * 7: Myanmar
         * 8: Philippines
         * 9: Singapore
         * 10: Thailand
         * 11: Viet Nam
         * 12: China
         * 13: Japan 
         * 14: Korea, Rep.
         * 15: India
         * 16: Uzbakistan
         * 17: Qatar 
         * 18: United States
         * 19: United Kingdom
         * 20: France
         * 21: Russian Federation
         * 22: Germany
         * 23: Australia
         * 24: Hungary
         * 25: Ireland
        
        
         '''
         )
    if att_regn == 1:
        att_regn_1 = 1
        att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25  = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Brunei Darussalam',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Brunei Darussalam',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Brunei Darussalam')
        st.pyplot(fig)
    elif att_regn == 2: 
        att_regn_2 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Cambodia',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Cambodia',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Cambodia')
        st.pyplot(fig)
    elif att_regn == 3: 
        att_regn_3 = 1
        att_regn_1 = att_regn_2 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Timor-Leste',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Timor-Leste',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Timor-Leste')
        st.pyplot(fig)
    elif att_regn == 4: 
        att_regn_4 = 1
        att_regn_1 = att_regn_2= att_regn_3 =  att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Indonesia',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Indonesia',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Indonesia')
        st.pyplot(fig)
    elif att_regn == 5: 
        att_regn_5 = 1
        att_regn_1 = att_regn_2= att_regn_3 = att_regn_4 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Lao PDR',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Lao PDR',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Lao PDR')
        st.pyplot(fig)
    elif att_regn == 6: 
        att_regn_6 = 1
        att_regn_1 = att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Malaysia',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Malaysia',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Malaysia')
        st.pyplot(fig)
    elif att_regn == 7: 
        att_regn_7 = 1
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Myanmar',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Myanmar',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Myanmar')
        st.pyplot(fig)
    elif att_regn == 8: 
        att_regn_8 = 1
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Philippines',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Philippines',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Philippines')
        st.pyplot(fig)
    elif att_regn == 9: 
        att_regn_9 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_2 = att_regn_10 = att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Singapore',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Singapore',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Singapore')
        st.pyplot(fig)
    elif att_regn == 10: 
        att_regn_10 = 1
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 =  att_regn_11 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Thailand',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Thailand',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Thailand')
        st.pyplot(fig)
    elif att_regn == 11: 
        att_regn_11 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_12 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Viet Nam',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Viet Nam',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Viet Nam')
        st.pyplot(fig)
    elif att_regn == 12: 
        att_regn_12 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'China',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'China',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of China')
        st.pyplot(fig)
    elif att_regn == 13: 
        att_regn_13 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Japan',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Japan',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Japan')
        st.pyplot(fig)
    elif att_regn == 14: 
        att_regn_14 = 1    
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Korea, Rep.',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Korea, Rep.',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Korea, Rep.')
        st.pyplot(fig)
    elif att_regn == 15: 
        att_regn_15 = 1 
        att_regn_1 = att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'India',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'India',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of India')
        st.pyplot(fig)
    elif att_regn == 16: 
        att_regn_16 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Uzbakistan',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Uzbakistan',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Uzbakistan')
        st.pyplot(fig)
    elif att_regn == 17: 
        att_regn_17 = 1 
        att_regn_1 = att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Qatar',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Qatar',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Qatar')
        st.pyplot(fig)
    elif att_regn == 18: 
        att_regn_18 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'United States',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'United States',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of United States')
        st.pyplot(fig)
    elif att_regn == 19: 
        att_regn_19 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'United Kingdom',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'United Kingdom',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of United Kingdom')
        st.pyplot(fig)
    elif att_regn == 20: 
        att_regn_20 = 1 
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'France',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'France',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of France')
        st.pyplot(fig)
    elif att_regn == 21: 
        att_regn_21 = 1    
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_22 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Russian Federation',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Russian Federation',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Russian Federation')
        st.pyplot(fig)
    elif att_regn == 22: 
        att_regn_22 = 1    
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_23 = att_regn_24 = att_regn_25 = 0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Germany',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Germany',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Germany')
        st.pyplot(fig)
    elif att_regn == 23: 
        att_regn_23 = 1  
        att_regn_1 = att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_24 = att_regn_25 = 0   
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Australia',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Australia',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Australia')
        st.pyplot(fig)
    elif att_regn == 24:
        att_regn_24 = 1
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_25 = 0 
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Hungary',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Hungary',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Hungary')
        st.pyplot(fig)
    else:
        att_regn_25 = 1
        att_regn_1 = att_regn_2 =  att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = att_regn_12 =  att_regn_13 = att_regn_14 = att_regn_15 = att_regn_16 = att_regn_17 = att_regn_18 = att_regn_19 = att_regn_20 = att_regn_21 = att_regn_22 = att_regn_23 = att_regn_24 =0
        data = get_dataset()
        data_country = data.loc[data['Country '] == 'Ireland',["Unemployment"]]
        data_year = data.loc[data['Country '] == 'Ireland',["Year"]]
        df_unemploy = pd.concat([data_year, data_country], axis=1)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('Year')
        ax.set_ylabel('Unemployment')
        ax.plot(df_unemploy['Year'], df_unemploy['Unemployment'])
        ax.set_title('Unemployment of Ireland')
        st.pyplot(fig)
    
    user_input = np.array([att_infla,att_exch,att_money,att_gdp,att_fdi,att_regn_1,att_regn_2, att_regn_3,
                       att_regn_4, att_regn_5, att_regn_6, att_regn_7, 
                       att_regn_8, att_regn_9, att_regn_10, att_regn_11, att_regn_12, att_regn_13, att_regn_14, att_regn_15,
                       att_regn_16, att_regn_17, att_regn_18, att_regn_19, att_regn_20, att_regn_21, att_regn_22, att_regn_23, att_regn_24,att_regn_25]).reshape(1,-1)
        
    if st.sidebar.button('Estimate Unemployment'):
        data = get_dataset()
        data['Country '] = data['Country '].astype('category')
        data_final = pd.concat([data,pd.get_dummies(data['Country '], prefix='Country ')], axis=1).drop(['Country '],axis=1)
    
        y = data_final["Unemployment"]
        X = data_final.drop(['Year','Unemployment'], axis=1)
        #X = data_final
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    
        gbm_opt = GradientBoostingRegressor(learning_rate=0.01, n_estimators=500,
                                        max_depth=5, min_samples_split=10, 
                                        min_samples_leaf=1, subsample=0.7,
                                        max_features=28, random_state=101)
        gbm_opt.fit(X_train,y_train)
        predictions = gbm_opt.predict(X_test)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=X_test, mode='markers', name='Prediction'))
        fig.update_layout(title='Scatter Plot of Y_Test vs Prediction',
                  xaxis_title='Y_Test',
                  yaxis_title='X_test')
        st.plotly_chart(fig)
    
        #making a prediction
        gbm_predictions = gbm_opt.predict(user_input)
        st.write('The estimated Unemployment is: ', gbm_predictions)
    