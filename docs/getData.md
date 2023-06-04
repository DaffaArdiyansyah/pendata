# MengLoad Data dari File CSV dan Database

## Hal - hal yang perlu disiapkan ##

* Power BI
* Excel
* DataBase Mysql
* Database PostgrasSql
* File CSV



## Cara Load Data File CSV ke Excel

   Buka Excel, Kemudian Klik Data, Kemudian pilih Get Data. Selanjutnya pilih From File dan pilih File CSV.
   <img src="ss/csvtoexcel.png">
   Kemudian Cari file Csv yang disimpan di Komputer dan kemudian pilih dan Klik Load. 
   <img src="ss/excelloadcsv.png">

## Cara  Load Data Dari MySql DataBase Ke Power BI

   Pertama pastikan Mysql sudah berjalan dan File nya sudah ada di databasenya, Kemudian buka Power BI kemudian Klik panel Home pada menu di atas. Setela itu klik Get data kemudian klik More. Kemudian Klik DataBase dan pilih MYSQL Database dan klic Connect. Setelah itu pada inputan server isikan "localhost:3306" dan pada Database masukkan nama Database yang berisikan file yang ingin di load, Pada Saya bernama "pendatairis". Sebelum itu apabila belum instal MySQl Connector NET wajib menginstal Terlebih dahulu dan Mengulang Langkah Langkah di atas. Selanjutnya Jika telah diisi semua klik ok dan kemudian muncul navigator, pada halaman ini centang database yang mau di load.
<img src="ss/mysqltobi.png">
Kemudian Klik Load.

## Cara Load Data Dari PostgraSQL local

   Pertama pastikan postgrasql sudah berjalan dan File nya sudah ada di databasenya, Kemudian buka Power BI kemudian Klik panel Home pada menu di atas. Setela itu klik Get data kemudian klik More. Kemudian Klik DataBase dan pilih PostgraSQL Database dan klik Connect.  Setelah itu pada inputan server isikan "localhost:5432" dan pada Database masukkan nama Database yang berisikan file yang ingin di load, Pada Saya bernama "databaseiris". Selanjutnya Jika telah diisi semua klik ok dan kemudian muncul navigator, pada halaman ini centang database yang mau di load.
<img src="ss/posrgraslocal.png">
Kemudian Klik Load.

## Cara Load Data Dari PostgraSQL Cloud Server

   Pertama bikin akun di https://www.elephantsql.com/, Kemudian buat New Instance dan isikan formnya setelah dibuat database cloud servis di newinstance kemudian hubungan Pgadmin dengan Cloud server tadi, Dan Masukkan file nya ke data basenya..  pastikan postgrasql sudah berjalan dan File nya sudah ada di databasenya, Kemudian buka Power BI kemudian Klik panel Home pada menu di atas. Setela itu klik Get data kemudian klik More. Kemudian Klik DataBase dan pilih PostgraSQL Database dan klik Connect.  Setelah itu pada inputan server isikan "rosie.db.elephantsql.com:5432" dan pada Database masukkan nama Database Cloud Server yang berisikan file yang ingin di load, Pada Saya bernama "jwannlep" Nama server dan Nama Databse dapat dilihat pada Detail Di ElephantSQl di Browser.Kemudian apabila di suruh masukkan pasword pilih databse dan masukkan password yang di cantumkan pada elepehntsql. Selanjutnya Jika telah diisi semua klik ok dan kemudian muncul navigator, pada halaman ini centang database yang mau di load.
<img src="ss/posrgrascloud.png">
Kemudian Klik Load.



