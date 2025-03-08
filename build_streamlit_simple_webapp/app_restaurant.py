import streamlit as st
import time
import pandas as pd
from random import randint

# Title displayed on the streamlit web app
st.title(f"""    🍔 MY HORRIBLE MENU""")
st.write("-----")

# Menu Selection
st.markdown('### Main course:')
meals=['Hamburger','Sandwich','Pasta']
meal=st.selectbox('Choose one:', meals)

# Menu Selection
st.markdown('### Drinks:')
drinks=['Milkshake','Soda','Coffee']
drink=st.selectbox('Choose one:', drinks)

# Menu Selection
st.markdown('### Dessert:')
desserts=['Cake','Ice cream','Cookies']
dessert=st.selectbox('Choose one:', desserts)
st.write("-----")

# Add button that prints out the selected items
if st.button('Confirm order'):
    st.write(':blue[Processing your order...]')
    time.sleep(2) # Faking a processing time
    st.write(f'Order #{randint(1230, 1450)}. Preparing at the kitchen:', 
             meal, ',', 
             drink, 'and', 
             dessert)
    st.feedback("faces")
