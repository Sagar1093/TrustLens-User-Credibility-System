import pandas as pd
import numpy as np
from pydantic import BaseModel,Field,computed_field
from typing import Literal,Annotated
import pickle
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse,StreamingResponse
import io
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

with open("IsolationModel","rb") as x:
    model = pickle.load(x)

with open("scaler.pkl","rb") as y:
    scaler = pickle.load(y)

with open("score_scaler.pkl","rb") as z:
    score_scaler = pickle.load(z)


app = FastAPI()



    
scoring_features = ["Product_Rating","Frequency_of_Purchase",
               "Customer_Satisfaction","value_per_visit","return_ratio"]
    

class UserInput(BaseModel):
    Purchase_Amount:Annotated[float, Field(...,ge= 0,description="Price of the Product")]
    Frequency_of_Purchase:Annotated[float, Field(...,ge = 0,description="Frequency of purchase per month")]
    Product_Rating:Annotated[int, Field(...,ge = 0,le = 5,description="Rating of Product")]
    #Time_Spent_on_Product_Research_hours:Annotated[float, Field(...,ge = 0,description="Time Taken to Research about a product")]
    Total_Orders:Annotated[int, Field(...,ge = 0,description="Total orders by a user")]
    Returned_Orders:Annotated[int, Field(...,ge = 0,description="Returned orders by a user")]
    Customer_Satisfaction:Annotated[float, Field(...,ge = 0,description="Overall user satisfaction")]

    @computed_field
    @property
    def Return_Rate(self) -> float:
        r = (self.Returned_Orders/(self.Total_Orders+1))*10
        return r
    
    @computed_field
    @property
    def value_per_visit(self) -> float:
        return (self.Purchase_Amount / 5) / (self.Frequency_of_Purchase + 1)

    @computed_field
    @property
    def return_ratio(self) -> float:
        return self.Return_Rate/(self.Frequency_of_Purchase+1)


@app.post("/predict")
def predict_anomaly(data: UserInput):

    df = pd.DataFrame([{
        "Product_Rating":data.Product_Rating,
        "Frequency_of_Purchase":data.Frequency_of_Purchase,
        "Customer_Satisfaction":data.Customer_Satisfaction,
        "value_per_visit":data.value_per_visit,
        "return_ratio":data.return_ratio
    }])

    features = df[["Product_Rating","Frequency_of_Purchase",
               "Customer_Satisfaction","value_per_visit","return_ratio"]]
    
 
    
    scaled_features = scaler.transform(features)

    scores = model.decision_function(scaled_features)[0]

    prediction = model.predict(scaled_features)[0]

   
    
    
       
   
   

    row = df.iloc[0]

   

    base = (
        0.35 * row["Customer_Satisfaction"] +
        0.25 * row["Product_Rating"] +
        0.15 * row["Frequency_of_Purchase"] -
        0.1 * (1 - row["return_ratio"]) +
        0.10 * row["value_per_visit"]
    )

    
    base = base ** 1.2

    base = max(0, min(base, 1))
    
    if (
    row["Product_Rating"] <= 2 and
    row["Customer_Satisfaction"] <= 4 and
    data.Return_Rate >= 0.8 and 
    row["Frequency_of_Purchase"]<15
    ):
        prediction = 1

    
    if prediction == -1:
        base /= 5

    trust_score = base * 100


    if prediction == -1 and trust_score < 35 :
        label = "Low Trust"
    elif trust_score > 60:
        label = "High Trust"
    elif trust_score > 30:
        label = "Medium Trust"
    else:
        label = "Low Trust"

    prompt = f"""
    You are an ML explanation engine for an Isolation Forest based User Credibility System.

    Your task is to explain the prediction using ONLY the provided numerical data.

    Rules:

    * Do NOT invent emotions, intentions, psychology, honesty, fraud, or loyalty.
    * Do NOT assume why the customer behaved this way.
    * Do NOT create narratives.
    * Do NOT contradict the input values.
    * Keep explanations technical and data-grounded.
    * Explain statistical behavior only.
    * Isolation Forest detects abnormal patterns, not morality.
    
    Prediction Mapping:
    * 1 = Normal behavior
    * -1 = Anomalous behavior
    Input Data:
    * Product Rating: {row["Product_Rating"]}
    * Total Orders: {data.Total_Orders}
    * Returned Orders: {data.Returned_Orders}
    * Customer Satisfaction: {row["Customer_Satisfaction"]}
    * Trust Score: {trust_score}
    * Prediction: {prediction}
    Output Format:
    Trust Analysis
    * Explain product rating briefly.
    * Explain purchase frequency briefly.
    * Explain return behavior briefly.
    * Explain customer satisfaction briefly.
    * Explain whether the model detected normal or anomalous statistical behavior.
    * Mention trust score separately from anomaly prediction.
    Conclusion:
    Provide a short factual conclusion fully supported by the input data.
    IMPORTANT:
    Every statement must directly relate to the provided values.
    If something is not numerically supported, do not mention it.
    Additional Rules:

    Do NOT expose raw model labels like 1 or -1 in the explanation.
    Interpret them as:
    1 → statistically normal behavior
    -1 → statistically anomalous behavior
    Trust Score is independent from the Isolation Forest prediction.
    Do NOT imply that the model directly generated the trust score.
    Describe the trust score as a separate behavioral scoring metric.
    Do NOT calculate metrics manually.
    Use ONLY backend-provided computed values.
    Add new line when needed to make it look good
    """
    
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    ai_reason = response.text

    return JSONResponse(status_code=200,content={"Prediction:":int(prediction),
                                                 "Trust_score":float(trust_score),
                                                 "trust_label":label,
                                                 "Reason(AI)":ai_reason})
    
@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    contents = file.file.read()
    df = pd.read_csv(io.BytesIO(contents))

    df["value_per_visit"] = df["Purchase_Amount"]/(df["Frequency_of_Purchase"]+1)
    df["Return_Rate"] = (
    df["Returned_Orders"] /(df["Total_Orders"] + 1)
    )*10
    

    df["return_ratio"] = df["Return_Rate"]/(df["Frequency_of_Purchase"]+1)
    

    features = df[["Product_Rating","Frequency_of_Purchase",
               "Customer_Satisfaction","value_per_visit","return_ratio"]]

    df_scaled = df.copy()

    scaled_features = scaler.transform(features)

    df_scaled["anomaly"] = model.predict(scaled_features)    

   
    df_scaled['anomaly_score'] = model.decision_function(scaled_features)
   
    df_scaled[scoring_features] = score_scaler.transform(
    df_scaled[scoring_features]
)

# Apply row-wise trust score logic
    def calculate_trust(row):

        base = (
            0.35 * row["Customer_Satisfaction"] +
            0.25 * row["Product_Rating"] +
            0.15 * row["Frequency_of_Purchase"] -
            0.1 * (1 - row["return_ratio"]) +
            0.10 * (row["value_per_visit"])
        )

       
        base *= 0.67

        base = max(0, min(base, 1))

        # Genuine dissatisfied customer override
        if (
            row["Product_Rating"] <= 2 and
            row["Customer_Satisfaction"] <= 4 and
            row["Return_Rate"] >= 0.8 and
            row["Frequency_of_Purchase"] < 15
        ):
            row["anomaly"] = 1

        # Penalize anomaly users
        if row["anomaly"] == -1:
                base *= 0.35

        trust_score = round(base,2)

        return pd.Series([
                trust_score,
                row["anomaly"]
            ])

    df_scaled[["Trust_Score", "anomaly"]] = df_scaled.apply(
            calculate_trust,
            axis=1
        )

    
        

   

   
    df_scaled["Trust_Score"] = df_scaled["Trust_Score"]*100
    df_scaled["Trust_Score"] = df_scaled["Trust_Score"].round(2)
    df_scaled["anomaly_score"] = df_scaled["anomaly_score"].round(3)

    
    df["anomaly"] = df_scaled["anomaly"]
    df["Trust_Score"] = df_scaled["Trust_Score"]
   

    df["Customer_Satisfaction"]= df["Customer_Satisfaction"].round(3).astype(float)
    df["Product_Rating"]= df["Product_Rating"].round(3).astype(float)
    df["Frequency_of_Purchase"]= df["Frequency_of_Purchase"].round().astype(int)
    df["return_ratio"]= df["return_ratio"].round(3).astype(float)
    df["value_per_visit"]= df["value_per_visit"].round(3).astype(float)
    def label(row):

        if row["anomaly"] == -1 and row["Trust_Score"] < 40 :
            return "Low Trust"
        elif row["Trust_Score"] > 60:
            return "High Trust"
        elif row["Trust_Score"] > 30:
            return "Medium Trust"
        else:
            return "Low Trust"

    df["Trust_Lable"] = df.apply(label,axis =1)


    output = io.StringIO()

    df.to_csv(output,index= False)

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="type/csv",
        headers={
            "Content-Disposition":"attachment;filename = result.csv"

        }
    )

