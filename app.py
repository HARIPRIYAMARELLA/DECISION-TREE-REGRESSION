import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeRegressor

#Load Dataset
df = pd.read_csv("CarPrice_Assignment.csv")

#Encode categorical columns
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

#Features and target
X = df.drop('price',axis=1)
y = df['price']

#Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train-test split
X_train,X_test,y_train,y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

#Train model
model = DecisionTreeRegressor()
model.fit(X_train,y_train)

#Create pickle inside app
pickle.dump(model,open("decision_tree_regressor.pkl","wb"))

#Load pickle
loaded_model = pickle.load(
    open("decision_tree_regressor.pkl","rb")
)

st.title("Car Price Prediction")
st.write("Enter Car Details")

symboling = st.number_input("Symboling",0)
wheelbase = st.number_input("Wheelbase",88.6)
carlength = st.number_input("Car Length",168.8)
carwidth = st.number_input("Car Width",64.1)
carheight = st.number_input("Car Height",48.8)
curbweight = st.number_input("Curb Weight",2548)
enginesize = st.number_input("Engine Size",130)
boreratio = st.number_input("Bore Ratio",3.47)
stroke = st.number_input("Stroke",2.68)
compressionratio = st.number_input("Compression Ratio",9.0)
horsepower = st.number_input("Horse Power",111)
peakrpm = st.number_input("Peak RPM",5000)
citympg = st.number_input("City MPG",21)
highwaympg = st.number_input("Highway MPG",27)

if st.button("Predict Price"):

    input_data=[[

        symboling,
        wheelbase,
        carlength,
        carwidth,
        carheight,
        curbweight,
        enginesize,
        boreratio,
        stroke,
        compressionratio,
        horsepower,
        peakrpm,
        citympg,
        highwaympg

    ]]

    input_scaled=scaler.transform(input_data)

    prediction=loaded_model.predict(input_scaled)

    st.success(
        f"Predicted Price: ${prediction[0]:,.2f}"
    )