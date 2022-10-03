import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from pickle import load

df=pd.read_csv('data/diamonds.csv')

or_enc=load(open('models/ordinal_encoder.pkl','rb'))
scaler = load(open('models/Standard_scaler.pkl', 'rb'))
rf_regressor=load(open('models/dt_regressor.pkl','rb'))
st.title('Diamond Price Prediction')

with st.form('my_form'):
    # carat=st.select_slider('Carat', options=df.carat.unique())
    carat = st.number_input('Enter carat')
    cut=st.selectbox(label='Cut of Diamond', options=df.cut.unique())
    color=st.selectbox(label='Color of Diamond', options=df.color.unique())
    clarity=st.selectbox(label='Clarity level of Diamond', options=df.clarity.unique())
    
    depth = st.number_input('Enter Depth')
    table = st.number_input('Enter table')
    x = st.number_input('Select Length of diamond in mm')
    y = st.number_input('Width of diamond in mm')
    z = st.number_input('Depth of diamond in mm')

    btn = st.form_submit_button(label='Predict')

    if btn:
        if carat and cut and color and clarity and depth and table and x and y and z:
            query_num = pd.DataFrame({'carat':[carat], 'depth':[depth],'table':[table],'x':[x],'y':[y],'z':[z]})
            query_cat = pd.DataFrame({'cut':[cut], 'color':[color], 'clarity':[clarity]})   

            query_cat = or_enc.transform(query_cat)
            query_num = scaler.transform(query_num)

            query_point = pd.concat([pd.DataFrame(query_num), pd.DataFrame(query_cat)], axis=1)
            price = dt_regressor.predict(query_point)

            st.success(f"The price of Selected Diamond is $ {round(price[0],2)}")

        else:
            st.error('Please enter all values')
