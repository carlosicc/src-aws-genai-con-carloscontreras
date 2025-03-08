import streamlit as st

# Title
st.title(f"""        😃 :rainbow[Super App!]""")

# More at: docs.streamlit.io/develop/api-reference/widgets/st.button
if st.button('Muestra imagen'):
   simulated_function = 'Imaginemos que esto es el output de una funcion'
   st.write(simulated_function)
   image = st.image('assets/images/cat.jpg')
