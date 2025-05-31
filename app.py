
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import joblib

with open('final_model.joblib','rb') as file:
    model = joblib.load(file)
    
with open('transformer.pkl','rb') as file:
    transformer = pickle.load(file)
    
def prediction(input_list):
    input_list = np.array(input_list,dtype=object)
    
    pred = model.predict_proba([input_list])[:,1][0]
    
    if pred>0.5:
        return f'This booking is more likely to get canceled, chances {round(pred,2)}'
    else:
        return f'This booking is less likely to get canceled, chances {round(pred,2)}'
    
# def main():
#     st.title('INN HOTEL GROUP')
#     lt = st.text_input('Enter the lead time in days')
#     mkt = (lambda x:1 if x=='Online' else 0)(st.selectbox('How the booking was made',['Online','Offline']))
#     price = st.text_input('Enter the price of the room')
#     adult = st.selectbox('How many adults',[1,2,3,4])
#     arr_m = st.slider('What is the month of arrival?',min_value=1,max_value=12,step=1)
#     weekd_lambda= (lambda x: 0 if x=='Mon' else
#                              1 if x=='Tue' else
#                              2 if x=='Wed' else
#                              3 if x=='Thus' else
#                              4 if x=='Fri' else
#                              5 if x=='Sat' else 6)
#     arr_w = weekd_lambda(st.selectbox('What is the weekday of arrival',['Mon','Tue','Wed','Thus','Fri','Sat','Sun']))
#     dep_w = weekd_lambda(st.selectbox('What is the weekday of departure',['Mon','Tue','Wed','Thus','Fri','Sat','Sun']))
#     weekn = st.text_input('Enter the no of week nights in stay')
#     wkndn = st.text_input('Enter the no of weekend nights in stay')
#     totan = weekn + wkndn
#     park = (lambda x:1 if x=='yes' else 0)(st.selectbox('Does customer need parking',['yes','no']))
#     spcl = st.selectbox('How many special requests have been mode',[0,1,2,3,4,5])
    
#     lt_t,price_t = transformer.transform([[lt,price]])[0]
    
#     inp_list = [lt_t,spcl,price_t,adult,wkndn,park,weekn,mkt,arr_m,arr_w,totan,dep_w]
    
#     if st.button('Predict'):
#         response = prediction(inp_list)
#         st.success(response)




def main():
    st.title('INN HOTEL GROUP')
    
    # 1. Numeric Inputs (converted to correct type upfront)
    lt = st.number_input('Enter the lead time in days', min_value=0, value=0)
    price = st.number_input('Enter the price of the room', min_value=0.0, value=0.0)
    
    # 2. Categorical Inputs
    mkt = 1 if st.selectbox('How the booking was made', ['Online', 'Offline']) == 'Online' else 0
    adult = st.selectbox('How many adults', [1, 2, 3, 4])
    arr_m = st.slider('What is the month of arrival?', min_value=1, max_value=12, step=1)
    
    # Weekday mapping (simplified)
    weekday_map = {'Mon':0, 'Tue':1, 'Wed':2, 'Thus':3, 'Fri':4, 'Sat':5, 'Sun':6}
    arr_w = weekday_map[st.selectbox('Arrival weekday', list(weekday_map.keys()))]
    dep_w = weekday_map[st.selectbox('Departure weekday', list(weekday_map.keys()))]
    
    # 3. Numeric Inputs with validation
    weekn = st.number_input('Week nights', min_value=0, value=0)
    wkndn = st.number_input('Weekend nights', min_value=0, value=0)
    totan = weekn + wkndn  # Correctly sums numbers
    
    park = 1 if st.selectbox('Need parking?', ['yes', 'no']) == 'yes' else 0
    spcl = st.selectbox('Special requests', [0, 1, 2, 3, 4, 5])
    
    # 4. Safe Transformation
    try:
        lt_t, price_t = transformer.transform([[lt, price]])[0]
    except Exception as e:
        st.error(f"Error processing inputs: {str(e)}")
        st.stop()  # Prevents further execution
    
    # 5. Prepare final feature list
    inp_list = [lt_t, spcl, price_t, adult, wkndn, park, weekn, mkt, arr_m, arr_w, totan, dep_w]
    
    # 6. Prediction with button
    if st.button('Predict'):
        try:
            response = prediction(inp_list)
            st.success(response)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
        
if __name__=='__main__':
    main()
