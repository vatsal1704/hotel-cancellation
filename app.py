import numpy as np
import pandas as pd
import streamlit as st
import pickle
import joblib

# Load trained model and transformer
with open('final_model.joblib', 'rb') as file:
    model = joblib.load(file)

with open('transformer.pkl', 'rb') as file:
    transformer = pickle.load(file)

# Prediction function
def prediction(input_list):
    input_list = np.array(input_list, dtype=object)
    pred = model.predict_proba([input_list])[:, 1][0]

    if pred > 0.5:
        return f'This booking is more likely to get canceled, chances {round(pred, 2)}'
    else:
        return f'This booking is less likely to get canceled, chances {round(pred, 2)}'

# Streamlit UI
def main():
    st.title('INN HOTEL GROUP')

    # 1. Numeric Inputs using number_input for safety
    lt = st.number_input('Enter the lead time in days', min_value=0.0)
    price = st.number_input('Enter the price of the room', min_value=0.0)
    weekn = st.number_input('Week nights', min_value=0, step=1)
    wkndn = st.number_input('Weekend nights', min_value=0, step=1)

    # 2. Categorical Inputs
    mkt = 1 if st.selectbox('How the booking was made', ['Online', 'Offline']) == 'Online' else 0
    adult = st.selectbox('How many adults', [1, 2, 3, 4])
    arr_m = st.slider('What is the month of arrival?', min_value=1, max_value=12, step=1)

    weekday_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thus': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
    arr_w = weekday_map[st.selectbox('Arrival weekday', list(weekday_map.keys()))]
    dep_w = weekday_map[st.selectbox('Departure weekday', list(weekday_map.keys()))]

    park = 1 if st.selectbox('Need parking?', ['yes', 'no']) == 'yes' else 0
    spcl = st.selectbox('Special requests', [0, 1, 2, 3, 4, 5])

    # 3. Total nights
    totan = weekn + wkndn

    # 4. Apply transformation on lead time and price
    try:
        lt_t, price_t = transformer.transform([[lt, price]])[0]
    except Exception as e:
        st.error(f"Error processing inputs: {str(e)}")
        st.stop()

    # 5. Prepare input list
    inp_list = [lt_t, spcl, price_t, adult, wkndn, park, weekn, mkt, arr_m, arr_w, totan, dep_w]

    # 6. Prediction
    if st.button('Predict'):
        try:
            response = prediction(inp_list)
            st.success(response)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Run the app
if __name__ == '__main__':
    main()
