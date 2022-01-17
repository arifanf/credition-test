import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Credition",
                    page_icon="credit-card")

st.write("""
# Customer Payment Prediction

This page to predict **Customer Payment** by Credition.
""")

st.sidebar.header('Input Data')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
else:
        def user_input_features():
                motor = st.sidebar.selectbox('Tipe Motor',('REVO FIT FI', 'REVO CW FI', 'NEW SUPRA X 125 FI SW',
                        'NEW SUPRA X 125 FI CW', 'BEAT POP CBS ISS',
                        'NEW VARIO 125 CBS ISS', 'BEAT SPORTY CW', 'BEAT STREET CBS',
                        'BEAT SPORTY CBS ISS', 'BEAT SPORTY CBS', 'ALL NEW SCOOPY',
                        'VARIO 125 CBS', 'VARIO 125 CBS ISS', 'GENIO CBS', 'GENIO CBS ISS',
                        'ALL NEW SONIC 150 R', 'SUPRA GTR 150', 'CB150 VERZA SW',
                        'CB150 VERZA CW', 'ALL NEW CBR150R', 'NEW CBR 150R STD',
                        'NEW CBR 150R ABS', 'CB150R STREETFIRE STD',
                        'CB150R STREETFIRE SE', 'CRF150L', 'PCX 150 CBS', 'PCX 150 ABS',
                        'PCX HYBRID', 'VARIO 150', 'ADV150 CBS', 'ADV150 ABS',
                        'CBR 250RR STD', 'CBR 250RR ABS', 'FORZA', 'SUPER CUB C125',
                        'NEW VARIO 150'))
                warna = st.sidebar.selectbox('Warna', (
                        'HITAM', 'MERAH-HITAM', 'MERAH', 'PUTIH', 'COKLAT HITAM','MERAH-PUTIH',
                        'COKLAT', 'SILVER', 'BIRU', 'PUTIH HITAM', 'BIRU-PUTIH', 'MAGENTA HITAM',
                        'HITAM-MERAH', 'BIRU-HITAM', 'PUTIH-MERAH', 'HITAM-SILVER', 'PUTIH-BIRU',
                        'HITAM - COKLAT', 'WHITE RED', 'ORANGE-PUTIH ', 'WHITE BLUE', 'KREM SILVER'))
                jk = st.sidebar.selectbox('Jenis Kelamin', ('LAKI-LAKI', 'PEREMPUAN'))
                rumah = st.sidebar.selectbox('Status Rumah', ('RUMAH SENDIRI', 'RUMAH ORANG TUA / KELUARGA', 
                        'RUMAH SEWA'))
                pekerjaan = st.sidebar.selectbox('Pekerjaan', ('DOKTER', 'PEGAWAI SWASTA,', 'MAHASISWA', 
                        'WIRASWASTA', 'TNI POLRI', 'IBU RUMAH', 'PEGAWAI NEGERI', 'GURU DOSEN', 'PETANI', 
                        'PENGACARA', 'OJEK', 'NELAYAN','PEGAWAI SWASTA', 'LAIN-LAIN'))
                pengeluaran = st.sidebar.selectbox('Pengeluaran (minimal range)', (2000001, 3000001, 1000001,  700001, 
                                1500001,  700000, 4000000))
                merk = st.sidebar.selectbox('Merk Motor Sebelumnya', ('BELUM PERNAH MEMILIKI', 'HONDA', 'YAMAHA', 
                        'SUZUKI', 'KAWASAKI', 'MOTOR LAIN'))
                tipe = st.sidebar.selectbox('Tipe Motor Sebelumnya', ('BELUM PERNAH MEMILIKI', 'AT AUTOMATIC', 
                        'SPORT', 'CUB BEBEK'))
                fungsi = st.sidebar.selectbox('Kegunaan Motor', ('BERDAGANG', 'PEMAKAIAN JARAK DEKAT', 
                        'KEBUTUHAN KELUARGA', 'LAIN-LAIN', 'KE SEKOLAH/ KE KAMPUS', 'BEKERJA', 'REKREASI/ OLAH RAGA'))
                pengguna = st.sidebar.selectbox('Pengguna Motor', ('SAYA SENDIRI', 'ANAK', 'PASANGAN SUAMI ATAU ISTRI', 
                        'LAIN-LAIN'))
                hobi = st.sidebar.selectbox('Hobi', ('Adventure (Petualangan)', 'Makan', 'Membaca', 'Memancing',
                        'Melukis', 'Volley', 'Massage', 'Memasak', 'Menyanyi', 'Sepakbola',
                        'Chatting', 'Jogging', 'Mendengarkan Radio',
                        'Memelihara Binatang Peliharaan', 'Membaca Puisi', 'Bersepeda ',
                        'Otomotif', 'Menjahit', 'Mengoleksi Barang Antik', 'Shopping',
                        'Bercocok Tanam', 'Golf', 'Renang', 'Badminton', 'Mendongeng',
                        'Mengaji', 'Travelling', 'Menanam Bunga', 'Senam', 'Fotografi',
                        'Bermain Komputer', 'Menonton TV', 'Berkaraoke', 'Surat Menyurat',
                        'Menggambar', 'Basket', 'Menari', 'Bermain Drama',
                        'Browsing Internet', 'Menonton Film', 'Fitness',
                        'Mendengarkan Musik', 'Menulis Buku', 'Sepatu Roda',
                        'Koleksi Perangko (Fillateli)', 'Bermain Games',
                        'Mengarang Cerita', 'Yoga', 'Origami', 'Menonton Bioskop',
                        'Bermain Sulap ', 'Aeromodeling', 'Bowling', 'Bermain Musik',
                        'Pantomim', 'Tennis'))
                lahir = st.sidebar.slider('Tahun Lahir', 1920, 2022)

                data = {'TYPE MOTOR': motor,
                        'COLOR': warna,
                        'JENIS KELAMIN': jk,
                        'STATUS RUMAH': rumah,
                        'PEKERJAAN': pekerjaan,
                        'PENGELUARAN': pengeluaran,
                        'MERK MOTOR SBLMNYA': merk,
                        'TYPE MOTOR SBLMNYA': tipe,
                        'SMH DIGUNAKAN UNTUK': fungsi,
                        'YG MENGGUNAKAN SMH': pengguna,
                        'HOBI': hobi,
                        'TAHUN LAHIR': lahir
                        }
                features = pd.DataFrame(data, index=[0])
                return features

        input_df = user_input_features()

konsumen_raw = pd.read_csv('credit_cleaned.csv')
konsumen = konsumen_raw.drop(columns=["JENIS PENJUALAN STNK"])
df = pd.concat([input_df,konsumen],axis=0)

encode = ["TYPE MOTOR", "COLOR", "JENIS KELAMIN", "STATUS RUMAH", "PEKERJAAN", 
            "MERK MOTOR SBLMNYA", "TYPE MOTOR SBLMNYA", "SMH DIGUNAKAN UNTUK", 
            "YG MENGGUNAKAN SMH", "HOBI"]
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] 

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_pd = pickle.load(open('credit.pkl', 'rb'))

# Apply model to make predictions
prediction = load_pd.predict(df)
prediction_proba = load_pd.predict_proba(df)

st.subheader('Prediction')
payment = np.array(['CASH','CREDIT'])
st.write(payment[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.caption("*This app Developed By <br>**Tim 2***")
