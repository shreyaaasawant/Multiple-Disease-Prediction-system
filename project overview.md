# Project Overview

## Project Name
Multiple Disease Prediction System

## Purpose

The Multiple Disease Prediction System is an AI-powered healthcare web application built using Streamlit, Machine Learning, and Google Gemini AI.

The application predicts the likelihood of multiple diseases using trained machine learning models and provides AI-assisted healthcare guidance.

This project is designed for educational and research purposes.

---

## Current Diseases Supported

- Diabetes
- Heart Disease
- Breast Cancer

---

## Core Features

### Disease Prediction
- Predicts diseases using trained ML models (.pkl)

### AI Health Assistant
- Google Gemini integration
- Explains prediction results
- Answers healthcare-related questions
- Provides personalized recommendations

### Medical Report Analysis
- Upload PDF medical reports
- Extract important health information
- Summarize reports using Gemini AI

### Personalized Precautions
- Diet suggestions
- Lifestyle improvements
- Exercise recommendations
- General preventive measures

### Email Reports
- Sends prediction reports
- Includes AI-generated recommendations
- PDF attachment support

---

## Technology Stack

Frontend
- Streamlit

Backend
- Python

Machine Learning
- Scikit-learn
- Pickle Models

AI
- Google Gemini API

Libraries
- Streamlit
- streamlit-option-menu
- FPDF
- smtplib
- google-generativeai
- pickle

---

## Current Project Structure

app.py
Main Streamlit application

heart.pkl
Heart Disease Model

diabetes.pkl
Diabetes Prediction Model

breast.pkl
Breast Cancer Model

requirements.txt
Dependencies

README.md
Documentation

---

## Future Vision

The goal is to transform this educational project into a professional AI healthcare assistant with modern UI and enhanced user experience while preserving all existing functionality.