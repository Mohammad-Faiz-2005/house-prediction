import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# from sklearn.naive_bayes import GaussianNB , MultinomialNB , BernoulliNB

st.title("house price prediction")

dataset = pd.read_csv(r"C:\Users\Mohammad Faiz\Downloads\dataset for datascience\Housing.csv")
df = pd.DataFrame(dataset)

x = df.iloc[:,:-1]
y = df["price"]

ss = StandardScaler()
ss.fit(x)
p = ss.transform(x)

x_train , x_test , y_train , y_test = train_test_split(p , y , test_size = 0.2 , random_state = 12)

dtr = RandomForestRegressor(    n_estimators=200,max_depth=15,min_samples_split=5,min_samples_leaf=2,random_state=12)
dtr.fit(x_train , y_train)
a = dtr.score(x_test , y_test )*100

st.write("Accuracy of decision Tree is: ",a)

bedroom = st.number_input("Enter no of bedrooms you required: ")
bathroom = st.number_input("Enter no of bath rooms you required: ")
sqft_living = st.number_input("Enter no of sqft_living you required: ")
sqft_lot = st.number_input("Enter no of sqft_lot you required: ")
floors = st.number_input("Enter no of floors you required: ")
condition = st.number_input("Enter condition you required: ")
# yearBuilt = st.number_input("Enter in which year builted house you accept: ")
grade = st.number_input("Enter grade you required: ")
zipcode = st.number_input("Enter zipcode you required: ")

b = dtr.predict([[bedroom , bathroom , sqft_living , sqft_lot , floors , condition , grade , zipcode ]]) #, yearBuilt]])
if st.button("Predict Price"):
    user_input = [[bedroom, bathroom, sqft_living, sqft_lot, floors, condition, grade, zipcode]]
    user_input_scaled = ss.transform(user_input)
    b = dtr.predict(user_input_scaled)
    st.write("Predicted price according to your input:", b)

