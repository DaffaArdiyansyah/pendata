#Library yang dibutuh kan
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Melakukan Pembacaan Judul (Introduction)
st.title("Aplikasi Prediksi Kanker Payudara")
st.write("##### NAMA : Daffa Ardiyansyah")
st.write("##### NIM  : 210411100134")
st.write("##### KELAS : PENAMBANGAN DATA B ")


# Tampilan Aturan Navbarnya 
data, preprocessing, modeling, implementasi = st.tabs([ "Data", "Preprocessing", "Modeling", "Implementasi"])

#Data Yang digunakan
df = pd.read_csv('https://raw.githubusercontent.com/DaffaArdiyansyah/pendata/main/docs/Breast_cancer_data.csv')


#Data
with data:
    st.header("Tentang Dataset")
    st.subheader("Aplikasi :")
    st.markdown("Aplikasi ini Digunakan untuk melakukan prediksi untuk penyakit kanker payudara dengan cara menginputkan parameter mean radius, mean texture, mean perimeter, mean area,Dan mean smoothness")
    
    st.subheader("Sumber Data :")
    st.markdown("Data di ambil dari website open data yaitu kaggle")
    st.markdown("Link : https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset")

    st.subheader("Tipe Data :")
    st.markdown("tipe Data pada dataset merupakan tipe data numerik, dan untuk labelnya (diagnosis) tipe data biner")
    st.dataframe(df)

    st.subheader("Tentang Data :")
    st.markdown("Di seluruh dunia, kanker payudara adalah jenis kanker yang paling umum pada wanita dan tertinggi kedua dalam hal angka kematian. Diagnosis kanker payudara dilakukan ketika ditemukan benjolan abnormal (dari pemeriksaan sendiri atau rontgen) atau setitik kecil dari kalsium terlihat (pada x-ray). Setelah benjolan yang mencurigakan ditemukan, dokter akan melakukan diagnosa untuk menentukan apakah itu kanker dan, jika demikian, apakah sudah menyebar ke bagian tubuh yang lain. Dataset kanker payudara ini dibuat dari University of Wisconsin Hospitals, Madison dari Dr. William H. Wolberg.")    


# Preprocessing
with preprocessing:
    st.subheader("Normalisasi Data")
    
    #data yang tidak digunakan dalam normalisasi yaitu kolom "diagnosis"
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"].values
    df_min = X.min()
    df_max = X.max()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    feature_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=feature_names)

    st.subheader("Normalisasi Data Diagnosa kenker payudara dengan MinMaxSxaler")
    st.write(scaled_features)



# Modeling
with modeling:

    training, test, training_label, test_label = train_test_split(scaled_features, y, test_size=0.2, random_state=42)
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilih Model untuk menghitung akurasi:")
        ann = st.checkbox('ANN')
        naive_bayes = st.checkbox('Naive Bayes')
        knn = st.checkbox('K-Nearest Neighbors')
        dt = st.checkbox('Decision Tree')
        submitted = st.form_submit_button("Submit")

        # ANN
        mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=443, random_state=42)
        mlp.fit(training, training_label)
        mlp_pred = mlp.predict(test)
        mlp_accuracy = round(100 * accuracy_score(test_label, mlp_pred))

        # Naive Bayes
        nb = GaussianNB()
        nb.fit(training, training_label)
        nb_pred = nb.predict(test)
        nb_accuracy = round(100 * accuracy_score(test_label, nb_pred))

        # KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(training, training_label)
        knn_pred = knn.predict(test)
        knn_accuracy = round(100 * accuracy_score(test_label, knn_pred))

        # Decision Tree
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(training, training_label)
        dt_pred = dt.predict(test)
        dt_accuracy = round(100 * accuracy_score(test_label, dt_pred))

        if submitted:
            if ann:
                st.write("Model ANN accuracy score: {0:0.2f}".format(mlp_accuracy))
            if naive_bayes:
                st.write("Model Naive Bayes accuracy score: {0:0.2f}".format(nb_accuracy))
            if knn:
                st.write("Model K-Nearest Neighbors accuracy score: {0:0.2f}".format(knn_accuracy))
            if dt:
                st.write("Model Decision Tree accuracy score: {0:0.2f}".format(dt_accuracy))

        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi': [mlp_accuracy, nb_accuracy, knn_accuracy, dt_accuracy],
                'Model': ['ANN', 'Naive Bayes', 'KNN','Decision Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

 

# Implementasi
with implementasi :
 with st.form("Daffa_form"):
    st.subheader('IMPLEMENTASI PREDIKSI KANKER PAYUDARA)')
    st.write("Inputkan nilai-nilai yang akan diprediksi:")  
    mean_radius  = st.number_input('Inputkan nilai mean radius') # sesuai data set
    mean_texture  = st.number_input('Inputkan nilai mean texture')
    mean_perimeter  = st.number_input('Inputkan nlai mean perimeter')
    mean_area  = st.number_input('Inputkan nilai mean area')
    mean_smoothness  = st.number_input('Inputkan nilai mean smoothness')
   


    model = st.selectbox('Pilih model untuk melakukan prediksi:',
                         ('ANN', 'Naive Bayes', 'KNN', 'Decision Tree'))

    prediksi = st.form_submit_button("Submit")
    if prediksi:
        inputs = np.array([mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness]) # atur sesuai variabelinputan data set
        input_norm = (inputs - df_min) / (df_max - df_min)
        input_norm = np.array(input_norm).reshape(1, -1)

        mod = None
        if model == 'ANN':
            mod = mlp
        elif model == 'Naive Bayes':
            mod = nb
        elif model == 'KNN':
            mod = knn
        elif model == 'Decision Tree':
             mod = dt

        if mod is not None:
            input_pred = mod.predict(input_norm)

            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan:', model)

            # code untuk prediksi
            cancer_diagnosis = ['']
        
            if (input_pred[0] == 0): 
                cancer_diagnosis = 'Pasien tidak Terkena Kanker Payudara'
            else :
                cancer_diagnosis = 'Terkena Kanker Payudara'

            st.success(cancer_diagnosis)
        else:
            st.write ('Model belum dipilih')