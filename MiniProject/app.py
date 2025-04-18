import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import base64
import math

# Set page config
st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="centered"
)

# Title and description
st.title("Loan Approval Prediction System")
st.write("Enter your details to check if your loan would be approved.")

# Function to format currency with Indian style commas (e.g., 12,00,000)
def format_indian_currency(amount):
    s = str(amount)
    l = len(s)
    if l <= 3:
        return s
    elif l <= 5:
        return s[0:l-3] + ',' + s[l-3:]
    else:
        return format_indian_currency(s[0:l-3]) + ',' + s[l-3:]

# Function to parse Indian currency format back to integer
def parse_indian_currency(amount_str):
    # Remove all commas and convert to integer
    if amount_str:
        return int(amount_str.replace(',', ''))
    return 0

# Function to load the model and feature names
@st.cache_resource
def load_model_and_features():
    # Load the model
    if os.path.exists('logmodel.pkl'):
        with open('logmodel.pkl', 'rb') as file:
            model = pickle.load(file)
    else:
        st.error("Model file 'logmodel.pkl' not found!")
        return None, None
    
    # Load feature names if available
    feature_names = []
    if os.path.exists('feature_names.pkl'):
        with open('feature_names.pkl', 'rb') as file:
            feature_names = pickle.load(file)
    else:
        st.warning("Feature names file not found. Using default feature mapping.")
        # Define default feature names based on common preprocessing for this dataset
        feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                         'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                         'Credit_History', 'Property_Area_Rural', 'Property_Area_Semiurban', 
                         'Property_Area_Urban']
    
    return model, feature_names

# Generate download link for CSV
def get_csv_download_link(df, filename="loan_entries.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}"><img src="https://img.icons8.com/material-outlined/24/000000/download--v1.png"/></a>'
    return href

# Function to calculate interest rate influence on probability
def calculate_interest_influence(base_probability, interest_rate):
    # Default interest rate is considered as 10%
    default_rate = 10.0
    
    # If interest rate is default, no change to probability
    if interest_rate == default_rate:
        return base_probability
    
    # Calculate the influence factor (exponential relationship)
    # Higher interest rates increase approval probability
    influence_factor = math.exp(0.05 * (interest_rate - default_rate))
    
    # Apply the influence factor to base probability
    adjusted_probability = base_probability * influence_factor
    
    # Ensure probability stays between 0 and 1
    return min(max(adjusted_probability, 0.0), 1.0)

# Load the model and feature names
model, feature_names = load_model_and_features()

if model is None:
    st.stop()

# Display feature names in a collapsible section in the sidebar
with st.sidebar.expander("Model Expected Features", expanded=False):
    st.write(feature_names)

# Add interest rate slider to sidebar
st.sidebar.title("Bank Settings")
interest_rate = st.sidebar.slider(
    "Interest Rate (%)", 
    min_value=8.0, 
    max_value=24.0, 
    value=10.0, 
    step=0.5,
    help="Higher interest rates may increase loan approval probability"
)

# Display selected interest rate impact
if interest_rate == 10.0:
    st.sidebar.info("Standard interest rate (10%) selected. No impact on approval probability.")
elif interest_rate < 10.0:
    st.sidebar.info(f"Lower interest rate ({interest_rate}%) may decrease approval probability.")
else:
    if interest_rate <= 15.0:
        impact = "moderate"
    elif interest_rate <= 20.0:
        impact = "significant"
    else:
        impact = "substantial"
    
    st.sidebar.success(f"Higher interest rate ({interest_rate}%) will have a {impact} positive impact on approval probability.")

# Initialize session state for multiple entries
if 'entries' not in st.session_state:
    st.session_state.entries = []

# Create form for user input
with st.form("loan_prediction_form"):
    st.subheader("Personal Information")
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        married = st.selectbox("Marital Status", ["Yes", "No"])
    
    col1, col2 = st.columns(2)
    with col1:
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    with col2:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    
    col1, col2 = st.columns(2)
    with col1:
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    with col2:
        credit_history = st.selectbox(
            "Credit History", 
            ["1", "0", "No History"], 
            help="1: Has good credit history, 0: Has bad credit history, No History: No previous credit record"
        )
        # Add red color to the help text for credit history
        st.markdown(
            """
            <style>
            .stSelectbox:nth-of-type(6) div[data-baseweb="tooltip"] {
                color: red !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    
    st.subheader("Financial Information")
    
    col1, col2 = st.columns(2)
    with col1:
        # Use text input for Indian currency format
        applicant_income_str = st.text_input("Annual Applicant Income (₹)", value="5,00,000", 
                                          help="Enter amount with commas (e.g., 12,00,000)")
        # Parse the currency string to get the numeric value
        try:
            applicant_income = parse_indian_currency(applicant_income_str)
        except:
            st.error("Please enter a valid amount in the format: 12,00,000")
            applicant_income = 500000
    
    with col2:
        # Use text input for Indian currency format
        coapplicant_income_str = st.text_input("Annual Co-applicant Income (₹)", value="0", 
                                            help="Enter amount with commas (e.g., 12,00,000)")
        # Parse the currency string to get the numeric value
        try:
            coapplicant_income = parse_indian_currency(coapplicant_income_str)
        except:
            st.error("Please enter a valid amount in the format: 12,00,000")
            coapplicant_income = 0
    
    col1, col2 = st.columns(2)
    with col1:
        # Use text input for Indian currency format
        loan_amount_str = st.text_input("Loan Amount (₹)", value="1,00,000", 
                                      help="Enter amount with commas (e.g., 12,00,000)")
        # Parse the currency string to get the numeric value
        try:
            loan_amount = parse_indian_currency(loan_amount_str)
        except:
            st.error("Please enter a valid amount in the format: 12,00,000")
            loan_amount = 100000
    
    with col2:
        loan_term = st.selectbox("Loan Term (months)", [360, 300, 240, 180, 120, 84, 60, 36, 12])
    
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    # Only add the Add Entry button inside the form
    # Center the "Add Entry" button
    col_center = st.columns([4, 2, 3])
    with col_center[1]:
        add_entry = st.form_submit_button("Add Entry")

# Handle adding entries
if add_entry:
    # Process credit history
    if credit_history == "No History":
        credit_history_value = 1  # Default to 1 as mentioned in the requirements
    else:
        credit_history_value = int(credit_history)
    
    entry = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history_value,
        'Property_Area': property_area,
        'Interest_Rate': interest_rate  # Store the interest rate with each entry
    }
    st.session_state.entries.append(entry)
    st.success(f"Entry added! Total entries: {len(st.session_state.entries)}")

col1, col2, col3, col4 = st.columns([2, 1, 2, 2])
with col1:
    if st.button("Clear All Entries"):
        st.session_state.entries = []
        st.session_state.show_table = False
        st.success("All entries cleared!")

with col4:
    submitted = st.button("Predict Loan Approval")



# --- Only proceed if entries exist ---
if st.session_state.entries:
    # Optional: set a flag if needed elsewhere
    st.session_state.show_table = True

    st.subheader("Entries for Prediction")

    # Create a DataFrame from entries
    entries_df = pd.DataFrame(st.session_state.entries)

    # Format currency columns with Indian style
    display_df = entries_df.copy()
    display_df['ApplicantIncome'] = display_df['ApplicantIncome'].apply(format_indian_currency)
    display_df['CoapplicantIncome'] = display_df['CoapplicantIncome'].apply(format_indian_currency)
    display_df['LoanAmount'] = display_df['LoanAmount'].apply(format_indian_currency)

    # Display table
    st.dataframe(display_df)

    # Add download CSV button
    st.markdown(get_csv_download_link(entries_df), unsafe_allow_html=True)

# Function to analyze reasons for approval/rejection
def analyze_approval_factors(input_data, prediction, prediction_proba, interest_rate):
    factors = {}
    reasons = []
    
    # Credit history is a major factor
    if input_data.get('Credit_History') == 1:
        factors["Credit History"] = "Good"
        reasons.append("You have a good credit history")
    elif input_data.get('Credit_History') == 0:
        factors["Credit History"] = "Poor"
        reasons.append("You have a poor credit history which significantly reduces approval chances")
    else:
        factors["Credit History"] = "No History"
        reasons.append("You have no credit history which makes assessment difficult")
    
    # Income assessment
    annual_income = input_data.get('ApplicantIncome') + input_data.get('CoapplicantIncome')
    if annual_income > 1000000:
        factors["Income Level"] = "High"
        reasons.append("Your combined annual income is high")
    elif annual_income > 500000:
        factors["Income Level"] = "Medium"
        reasons.append("Your combined annual income is average")
    else:
        factors["Income Level"] = "Low"
        reasons.append("Your combined annual income is relatively low")
    
    # Loan amount to income ratio
    loan_amount = input_data.get('LoanAmount')
    income_to_loan_ratio = annual_income / (loan_amount * 12) if loan_amount > 0 else float('inf')
    
    if income_to_loan_ratio < 2:
        factors["Loan-to-Income Ratio"] = "High"
        reasons.append("The loan amount is high compared to your income")
    elif income_to_loan_ratio < 5:
        factors["Loan-to-Income Ratio"] = "Moderate"
        reasons.append("The loan amount is reasonable compared to your income")
    else:
        factors["Loan-to-Income Ratio"] = "Low"
        reasons.append("The loan amount is very affordable compared to your income")
    
    # Property area
    property_area = input_data.get('Property_Area')
    factors["Property Area"] = property_area
    
    # Dependents
    dependents = input_data.get('Dependents')
    if dependents == "3+":
        dep_value = 3
    else:
        dep_value = int(dependents)
    
    if dep_value >= 2:
        factors["Dependents"] = f"{dependents} (High)"
        reasons.append("You have multiple dependents which increases financial responsibility")
    else:
        factors["Dependents"] = dependents
    
    # Interest rate effect
    factors["Interest Rate"] = f"{interest_rate}%"
    if interest_rate > 10.0:
        interest_impact = (interest_rate - 10.0) / 10.0 * 100  # Calculate percentage increase in probability
        reasons.append(
            f"Higher interest rate ({interest_rate}%) increases approval probability by approximately {interest_impact:.1f}% in odds.\n\nOdds = P / (1 - P)"
        )

    elif interest_rate < 10.0:
        interest_impact = (10.0 - interest_rate) / 10.0 * 100  # Calculate percentage decrease in probability
        reasons.append(f"Lower interest rate ({interest_rate}%) decreases approval probability by approximately {interest_impact:.1f}%")
    
    # Final conclusion
    if prediction == 1:
        if input_data.get('Credit_History') == 1 and income_to_loan_ratio > 2:
            conclusion = "Your good credit history and reasonable loan amount relative to income are key positive factors."
        elif input_data.get('Credit_History') == 1:
            conclusion = "Your approval is mainly due to your good credit history, though the loan amount is high relative to income."
        elif interest_rate > 15.0:
            conclusion = "The higher interest rate has significantly increased your approval chances despite other risk factors."
        else:
            conclusion = "Your strong income and reasonable loan amount overcame other negative factors."
    else:
        if input_data.get('Credit_History') == 0 and interest_rate < 15.0:
            conclusion = "Your application is likely to be rejected mainly due to your poor credit history. A higher interest rate might improve your chances."
        elif income_to_loan_ratio < 2 and interest_rate < 18.0:
            conclusion = "Your application is likely to be rejected because the loan amount is too high compared to your income. Consider a higher interest rate."
        else:
            conclusion = "Multiple factors contributed to the predicted rejection. Consider improving your credit history, requesting a lower loan amount, or accepting a higher interest rate."
    
    return factors, reasons, conclusion

# Prediction logic
if submitted:
    # Decide what to predict - either the form data or the entries
    entries_to_predict = st.session_state.entries if st.session_state.entries else [{}]
    
    # If no stored entries, use the form data
    if not st.session_state.entries:
        # Process credit history
        if credit_history == "No History":
            credit_history_value = 1  # Default to 1
        else:
            credit_history_value = int(credit_history)
            
        entries_to_predict = [{
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history_value,
            'Property_Area': property_area,
            'Interest_Rate': interest_rate  # Use current interest rate
        }]
    
    # Process each entry
    for idx, entry in enumerate(entries_to_predict):
        try:
            st.subheader(f"Prediction for Entry {idx+1}" if len(entries_to_predict) > 1 else "Prediction Result")
            
            # Get interest rate for this entry (or use current value if not stored)
            entry_interest_rate = entry.get('Interest_Rate', interest_rate)
            
            # Create a dictionary for user input
            input_dict = {}
            
            # Process categorical features to numeric based on common encodings
            input_dict['Gender'] = 1 if entry.get('Gender') == "Male" else 0
            input_dict['Married'] = 1 if entry.get('Married') == "Yes" else 0
            
            # Handle dependents
            dependents_val = entry.get('Dependents')
            if dependents_val == "3+":
                input_dict['Dependents'] = 3
            else:
                input_dict['Dependents'] = int(dependents_val)
                
            input_dict['Education'] = 1 if entry.get('Education') == "Graduate" else 0
            input_dict['Self_Employed'] = 1 if entry.get('Self_Employed') == "Yes" else 0
            
            # Divide ApplicantIncome by 100 as requested
            input_dict['ApplicantIncome'] = entry.get('ApplicantIncome') / 100
            input_dict['CoapplicantIncome'] = entry.get('CoapplicantIncome')
            
            # Divide LoanAmount by 1000 as requested
            input_dict['LoanAmount'] = entry.get('LoanAmount') / 1000
            input_dict['Loan_Amount_Term'] = entry.get('Loan_Amount_Term')
            input_dict['Credit_History'] = entry.get('Credit_History')
            
            # One-hot encode Property_Area
            property_area_val = entry.get('Property_Area')
            input_dict['Property_Area_Rural'] = 1 if property_area_val == "Rural" else 0
            input_dict['Property_Area_Semiurban'] = 1 if property_area_val == "Semiurban" else 0
            input_dict['Property_Area_Urban'] = 1 if property_area_val == "Urban" else 0
            
            # Create DataFrame with the exact same columns as training data
            if feature_names:
                # Create a new DataFrame with zeros for all features
                model_input = pd.DataFrame(0, index=[0], columns=feature_names)
                
                # Update with values from user input for features that exist in both
                for col in input_dict:
                    if col in feature_names:
                        model_input[col] = input_dict[col]
                
                # Replace input_df with properly structured DataFrame
                input_df = model_input
            else:
                input_df = pd.DataFrame([input_dict])
            
            # Make prediction
            base_prediction = model.predict(input_df)[0]
            base_probability = model.predict_proba(input_df)[0][1]
            
            # Adjust probability based on interest rate
            adjusted_probability = calculate_interest_influence(base_probability, entry_interest_rate)
            
            # Determine final prediction based on adjusted probability
            final_prediction = 1 if adjusted_probability > 0.5 else 0
            
            # Analyze factors and get reasons
            factors, reasons, conclusion = analyze_approval_factors(entry, final_prediction, adjusted_probability, entry_interest_rate)
            
            # Display result with comparison to base prediction if there's a difference
            if final_prediction == 1:
                st.success(f"✅ Congratulations! Your loan is likely to be APPROVED")
                
                # Show original vs adjusted probability if there's a significant difference
                if abs(adjusted_probability - base_probability) > 0.05:
                    st.write("Interest rate impact:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Base model probability:")
                        st.progress(float(base_probability))
                        st.write(f"{base_probability:.2%}")
                    with col2:
                        st.write("Adjusted with interest rate:")
                        st.progress(float(adjusted_probability))
                        st.write(f"{adjusted_probability:.2%}")
                else:
                    st.progress(float(adjusted_probability))
                    st.write(f"Approval probability: {adjusted_probability:.2%}")
            else:
                st.error(f"❌ Sorry, your loan is likely to be REJECTED")
                
                # Show original vs adjusted probability if there's a significant difference
                if abs(adjusted_probability - base_probability) > 0.05:
                    st.write("Interest rate impact:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Base model probability:")
                        st.progress(float(base_probability))
                        st.write(f"{base_probability:.2%}")
                    with col2:
                        st.write("Adjusted with interest rate:")
                        st.progress(float(adjusted_probability))
                        st.write(f"{adjusted_probability:.2%}")
                else:
                    st.progress(float(adjusted_probability))
                    st.write(f"Approval probability: {adjusted_probability:.2%}")
            
            # Display factors affecting decision
            st.subheader("Factors Affecting Decision")
            
            # Create two columns for factors and reasons
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Key Metrics:**")
                for factor, value in factors.items():
                    st.write(f"**{factor}:** {value}")
            
            with col2:
                st.write("**Analysis:**")
                for reason in reasons:
                    st.write(f"• {reason}")
            
            # Show conclusion
            st.info(f"**Summary:** {conclusion}")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please check that your model and input features match exactly.")
            import traceback
            st.code(traceback.format_exc())

# Add information in the sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This application predicts whether a loan would be approved based on "
    "personal and financial information. The model was trained on historical "
    "loan approval data with a logistic regression model."
)

# Additional tips
st.sidebar.title("Tips for Approval")
st.sidebar.markdown("""
- Maintain a good credit history
- Have a stable income source
- Request a reasonable loan amount relative to income
- Longer loan terms may increase approval chances
- Consider accepting a higher interest rate to increase approval chances
""")