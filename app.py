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
    
def main():
    st.title('INN HOTEL GROUP')
    lt = st.text_input('Enter the lead time in days')
    mkt = (lambda x:1 if x=='Online' else 0)(st.selectbox('How the booking was made',['Online','Offline']))
    price = st.text_input('Enter the price of the room')
    adult = st.selectbox('How many adults',[1,2,3,4])
    arr_m = st.slider('What is the month of arrival?',min_value=1,max_value=12,step=1)
    weekd_lambda= (lambda x: 0 if x=='Mon' else
                             1 if x=='Tue' else
                             2 if x=='Wed' else
                             3 if x=='Thus' else
                             4 if x=='Fri' else
                             5 if x=='Sat' else 6)
    arr_w = weekd_lambda(st.selectbox('What is the weekday of arrival',['Mon','Tue','Wed','Thus','Fri','Sat','Sun']))
    dep_w = weekd_lambda(st.selectbox('What is the weekday of departure',['Mon','Tue','Wed','Thus','Fri','Sat','Sun']))
    weekn = st.text_input('Enter the no of week nights in stay')
    wkndn = st.text_input('Enter the no of weekend nights in stay')
    totan = weekn + wkndn
    park = (lambda x:1 if x=='yes' else 0)(st.selectbox('Does customer need parking',['yes','no']))
    spcl = st.selectbox('How many special requests have been mode',[0,1,2,3,4,5])
    
    # lt_t,price_t = transformer.transform([[lt,price]])[0]
    
    # inp_list = [lt_t,spcl,price_t,adult,wkndn,park,weekn,mkt,arr_m,arr_w,totan,dep_w]
    
    # if st.button('Predict'):
    #     response = prediction(inp_list)
    #     st.success(response)
    
    
    
    
    # if st.button('Predict'):
    #     # Input validation
    #     if lt == '' or price == '' or weekn == '' or wkndn == '':
    #         st.error("Please fill in all required numerical fields.")
    #     else:
    #         # Convert values
    #         lt = float(lt)
    #         price = float(price)
    #         weekn = int(weekn)
    #         wkndn = int(wkndn)
    #         totan = weekn + wkndn
    #         adult = int(adult)
    #         spcl = int(spcl)
    
    #         # Prepare DataFrame for transformation
    #         input_df = pd.DataFrame([[lt, price]], columns=['lead_time', 'price'])
    
    #         # Transform
    #         lt_t, price_t = transformer.transform(input_df)[0]
    
    #         # Construct feature list
    #         inp_list = [lt_t, spcl, price_t, adult, wkndn, park, weekn, mkt, arr_m, arr_w, totan, dep_w]
    
    #         # Make prediction
    #         response = prediction(inp_list)
    #         st.success(response)
def main():
    st.title('INN HOTEL GROUP')
    
    lt = st.text_input('Enter the lead time in days')
    mkt = 1 if st.selectbox('How the booking was made', ['Online', 'Offline']) == 'Online' else 0
    price = st.text_input('Enter the price of the room')
    adult = int(st.selectbox('How many adults', [1, 2, 3, 4]))
    arr_m = st.slider('What is the month of arrival?', min_value=1, max_value=12, step=1)

    weekd_lambda = lambda x: {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thus': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}[x]
    arr_w = weekd_lambda(st.selectbox('What is the weekday of arrival', ['Mon', 'Tue', 'Wed', 'Thus', 'Fri', 'Sat', 'Sun']))
    dep_w = weekd_lambda(st.selectbox('What is the weekday of departure', ['Mon', 'Tue', 'Wed', 'Thus', 'Fri', 'Sat', 'Sun']))

    weekn = st.text_input('Enter the no of week nights in stay')
    wkndn = st.text_input('Enter the no of weekend nights in stay')

    park = 1 if st.selectbox('Does customer need parking', ['yes', 'no']) == 'yes' else 0
    spcl = int(st.selectbox('How many special requests have been made', [0, 1, 2, 3, 4, 5]))

    if st.button('Predict'):
        # Validate required fields
        if lt == '' or price == '' or weekn == '' or wkndn == '':
            st.error("Please fill in all required numerical fields.")
            return

        try:
            # Type conversions
            lt = float(lt)
            price = float(price)
            weekn = int(weekn)
            wkndn = int(wkndn)
            totan = weekn + wkndn

            # Transform lt and price
            input_df = pd.DataFrame([[lt, price]], columns=['lead_time', 'price'])
            lt_t, price_t = transformer.transform(input_df)[0]

            # Prediction input
            inp_list = [lt_t, spcl, price_t, adult, wkndn, park, weekn, mkt, arr_m, arr_w, totan, dep_w]
            response = prediction(inp_list)
            st.success(response)

        except ValueError:
            st.error("Invalid input. Please make sure all values are numeric where expected.")







        
if __name__=='__main__':
    main()
    st.success(response)

if __name__=='__main__':
    main()
