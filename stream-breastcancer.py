import pickle
import streamlit as st

# membaca model
breastcancer_model = pickle.load(open('naive_breastcancer.sav','rb'))


#judul web
st.title('Prediksi Kanker Payudara')

mean_radius = st.number_input ('input nilai mean radius')

mean_texture = st.number_input ('input nilai mean texture')

mean_perimeter =  st.number_input('input nilai mean perimeter')

mean_area = st.number_input ('input nilai mean area')

mean_smoothness = st.number_input ('input nilai mean smoothness')


# code untuk prediksi
cancer_diagnosis = ['']

#membuat tombol untuk presikdi
if st.button('Prediksi Kanker Payudara'):
    cancer_prediction = breastcancer_model.predict([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])

    if (cancer_prediction[0] == 0):
        cancer_diagnosis = 'Pasien tidak Terkena Kanker Payudara'
    else :
        cancer_diagnosis = 'Terkena Kanker Payudara'

    st.success(cancer_diagnosis)

