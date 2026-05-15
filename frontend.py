import pandas as pd
import numpy as np
import requests
import streamlit as st


st.title("User Credibilty System")

uploaded_file = st.file_uploader("Upload a CSV File",type = ["csv"])

if uploaded_file is not None:
    st.success("File Uploaded Successfully")

    df = pd.read_csv(uploaded_file)
    st.write("Preview of the file")
    st.dataframe(df.head())

    if st.button("Process file"):

        files = {
            "file":(uploaded_file.name,uploaded_file.getvalue(),"test/csv")
        }

        response = requests.post(
            "http://127.0.0.1:8000/predict_csv",
            files = files
        )
        if response.status_code == 200:
            st.success("Processing is completed")

            st.download_button(
                label = "Download Result CSV",
                data = response.content,
                file_name="Processed_result.csv",
                mime="text/csv"
            )

            
        else:
            st.error("API Error")


st.header("Input Demo Data")

purchace = st.number_input("Purchase Amount",min_value=0,value=300)
freq = st.number_input("Purchase Frequency(monthly)",min_value=0,value=3)
rating = st.slider("Rating of the Product",1,5,value= 3)
total_orders = st.number_input("Total orders by the User",min_value=0,value=15)
returned_orders= st.number_input("Returned orders by the User",min_value=0,value=2)
satis = st.slider("Customer Satisfaction",1,10,value = 5)
if st.button("predict"):

    payload = {
        "Purchase_Amount":purchace,
        "Frequency_of_Purchase":freq,
        "Product_Rating":rating,
        "Total_Orders":total_orders,
        "Returned_Orders":returned_orders,
        "Customer_Satisfaction":satis
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    if response.status_code == 200:

        result = response.json()

        prediction = result["Prediction:"]
        trust_score = result["Trust_score"]
        trust_label = result["trust_label"]
        ai_reason = result["Reason(AI)"]

        st.success("Prediction complete")

        st.subheader("Trust Analysis Dashboard")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Trust Score",
            f"{trust_score:.2f}/100"
        )

        col2.metric(
            "Trust Label",
            trust_label
        )

        col3.metric(
            "Prediction",
            prediction
        )

        st.progress(min(int(trust_score), 100))

        if trust_label == "High Trust":
            st.success("High Trust User")

        elif trust_label == "Medium Trust":
            st.warning("Medium Trust User")

        else:
            st.error("Low Trust User")

        st.markdown("---")

        st.subheader("AI Behavioral Analysis")

        st.markdown(
            f"""
            <div style="
                background-color:#111827;
                padding:20px;
                border-radius:15px;
                border-left:5px solid #00FF99;
                color:white;
                font-size:16px;
            ">
            {ai_reason}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("API Error")

    