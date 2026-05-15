# TrustLens - User-Credibility-System
User Credibility System for E-Commerce 

TrustLens – User Credibility System
Overview

TrustLens is a user credibility evaluation system that calculates a Trust Score based on behavioral and transactional data. It is designed to help digital platforms assess the reliability of users and support data-driven decision-making.

Problem Statement

Online platforms often face challenges such as fraudulent users, unreliable reviews, and lack of measurable trust indicators. TrustLens addresses these issues by providing a structured and scalable method to quantify user credibility.

Features
Trust Score calculation using weighted parameters
Data preprocessing and normalization
Visualization of results using graphs
Downloadable CSV output
REST API implementation
Interactive frontend interface
Technology Stack
Backend: FastAPI
Frontend: Streamlit
Data Processing: Pandas
Visualization: Matplotlib
Programming Language: Python
Trust Score Formula
Trust_Score = 
    0.35 * Customer_Satisfaction +
    0.25 * Product_Rating +
    0.15 * Frequency_of_Purchase -
    0.15 * (1 - Return_Rate) +
    0.10 * Review_Helpfulness

All input features are scaled before computing the final score.
Gemini 3.1 Flash is used here to generate an Explanation of the result


