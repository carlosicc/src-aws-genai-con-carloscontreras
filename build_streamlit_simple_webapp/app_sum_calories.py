import streamlit as st
import pandas as pd

# Datos del menú del restaurante
data = 'assets/data/menu_restaurante.csv'

# Read the CSV file
df = pd.read_csv(data)

# Mostrar el menú completo
st.subheader('🍔 Menú Completo 🍔')
st.table(df)

# Selección de alimentos
st.subheader('📝 Menu de hoy')
bebida = st.selectbox(' 🥛 Seleccione una bebida:', df[df['Categoría'] == 'Bebida']['Alimento'])
plato_principal = st.selectbox(' 🌮 Seleccione un plato principal:', df[df['Categoría'] == 'Plato Principal']['Alimento'])
postre = st.selectbox(' 🍰 Seleccione un postre:', df[df['Categoría'] == 'Postre']['Alimento'])

# Botón para calcular la cuenta
if st.button('Calcular Total'):
    # Filtrar el DataFrame para obtener los precios y calorías seleccionados
    total_bebida = df[df['Alimento'] == bebida]['Precio ($)'].values[0]
    total_plato_principal = df[df['Alimento'] == plato_principal]['Precio ($)'].values[0]
    total_postre = df[df['Alimento'] == postre]['Precio ($)'].values[0]

    calorias_bebida = df[df['Alimento'] == bebida]['Calorías'].values[0]
    calorias_plato_principal = df[df['Alimento'] == plato_principal]['Calorías'].values[0]
    calorias_postre = df[df['Alimento'] == postre]['Calorías'].values[0]

    # Calcular el total de precio y calorías
    total_precio = total_bebida + total_plato_principal + total_postre
    total_calorias = calorias_bebida + calorias_plato_principal + calorias_postre

    # Mostrar el total a pagar y las calorías totales
    st.write(f"**Total a pagar:** ${total_precio:.2f}")
    st.write(f"**Calorías totales:** {total_calorias} kcal")
    st.write(f"Has pedido: {bebida}, {plato_principal}, {postre}.")
