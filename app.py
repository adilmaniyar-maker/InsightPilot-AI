import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from reportlab.pdfgen import canvas
from io import BytesIO


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="InsightPilot AI",
    page_icon="📊",
    layout="wide"
)
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

h1 {
    color: #4F46E5;
}

[data-testid="stSidebar"] {
    background-color: #F3F4F6;
}
</style>
""", unsafe_allow_html=True)
st.sidebar.title("🔍 Filters")
st.title("📊 InsightPilot AI")
st.write("Upload a CSV or Excel file to begin.")

# -----------------------------
# Load Gemini API Key
# -----------------------------
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx"]
)

# -----------------------------
# Main Logic
# -----------------------------
if uploaded_file is not None:

    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File uploaded successfully!")
    # Sidebar Filters
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    if categorical_cols:
        filter_col = st.sidebar.selectbox(
            "Select Filter Column",
            ["None"] + categorical_cols
        )

        if filter_col != "None":
            options = df[filter_col].dropna().unique().tolist()

            selected = st.sidebar.multiselect(
                "Select Values",
                options,
                default=options
            )

            df = df[df[filter_col].isin(selected)]
    # -----------------------------
    # Metrics
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Duplicates", int(df.duplicated().sum()))

    # -----------------------------
    # Preview
    # -----------------------------
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Missing Values
    # -----------------------------
    st.subheader("❗ Missing Values")
    st.write(df.isnull().sum())

    # -----------------------------
    # Remove Duplicates
    # -----------------------------
    if st.button("Remove Duplicates"):
        df = df.drop_duplicates()
        st.success("Duplicates removed successfully!")

    # -----------------------------
    # Download Cleaned CSV
    # -----------------------------
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

    # -----------------------------
    # Dashboard
    # -----------------------------
    st.subheader("📊 Interactive Dashboard")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    #date_cols = df.select_dtypes(include=["datetime64", "datetime"]).columns.tolist()
    if numeric_cols:

        selected_col = st.selectbox(
            "Select Numeric Column",
            numeric_cols
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Total", f"{df[selected_col].sum():,.2f}")
        c2.metric("Average", f"{df[selected_col].mean():,.2f}")
        c3.metric("Maximum", f"{df[selected_col].max():,.2f}")

        fig = px.histogram(
            df,
            x=selected_col,
            title=f"Distribution of {selected_col}"
        )

        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(
            df,
            y=selected_col,
            title=f"Box Plot of {selected_col}"
        )

        st.plotly_chart(fig2, use_container_width=True)
    
    else:
        st.warning("No numeric columns found.")
    # -----------------------------
    # Correlation Heatmap
    # -----------------------------
    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] >= 2:

        corr = numeric_df.corr()

        fig3 = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix"
        )

        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("Need at least 2 numeric columns.")
    st.subheader("🥧 Pie Chart")

    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if cat_cols and numeric_cols:

        category = st.selectbox(
            "Category Column",
            cat_cols
        )

        value = st.selectbox(
            "Value Column",
            numeric_cols,
            key="pie"
        )

        pie_df = df.groupby(category)[value].sum().reset_index()

        fig4 = px.pie(
            pie_df,
            names=category,
            values=value,
            title=f"{value} by {category}"
        )

        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Need at least one categorical and one numeric column for pie chart.")
    # -----------------------------
    # Forecasting
    # -----------------------------
    st.subheader("📈 Simple Forecast")

    if numeric_cols:

        forecast_col = st.selectbox(
            "Select a numeric column for forecasting",
            numeric_cols,
            key="forecast_col"
        )

        series = df[forecast_col].dropna().reset_index(drop=True)

        if len(series) >= 5:

            # Create X and y
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values

            # Train model
            model = LinearRegression()
            model.fit(X, y)

            # Predict next 10 points
            future_X = np.arange(len(series) + 10).reshape(-1, 1)
            predictions = model.predict(future_X)

            # Create DataFrame
            forecast_df = pd.DataFrame({
                "Index": future_X.flatten(),
                "Predicted Value": predictions
            })

            # Plot
            fig_forecast = px.line(
                forecast_df,
                x="Index",
                y="Predicted Value",
                title=f"Forecast for {forecast_col}"
            )

            st.plotly_chart(fig_forecast, use_container_width=True)

        else:
            st.warning("Need at least 5 values for forecasting.")

    st.subheader("🏆 Top 10 Analysis")

    if numeric_cols:

        top_col = st.selectbox(
            "Select Column",
            numeric_cols,
            key="top10"
        )

        top_df = (
            df.sort_values(
                by=top_col,
                ascending=False
            )
            .head(10)
            .reset_index(drop=True)
        )

        fig_top = px.bar(
            top_df,
            x=top_df.index,
            y=top_col,
            title=f"Top 10 Records by {top_col}"
        )
        st.plotly_chart(fig_top, use_container_width=True)
        
    st.subheader("📉 Trend Analysis")

    if numeric_cols:

        trend_col = st.selectbox(
            "Select Trend Column",
            numeric_cols,
            key="trend"
        )

        fig_trend = px.line(
            df,
            y=trend_col,
            title=f"Trend of {trend_col}"
        )

        st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("🎯 Data Quality Score")

    total_cells = df.shape[0] * df.shape[1]

    missing = df.isnull().sum().sum()

    score = ((total_cells - missing) / total_cells) * 100

    st.metric(
        "Quality Score",
        f"{score:.2f}%"
    )

    st.subheader("📋 Dataset Profile")

    profile_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(profile_df)

    from io import BytesIO

    st.subheader("📊 Export Dataset Report")

    excel_buffer = BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)

    st.download_button(
        label="📥 Download Excel Report",
        data=excel_buffer.getvalue(),
        file_name="InsightPilot_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if st.button("Generate Executive Summary"):
        if not api_key:
            st.error("GEMINI_API_KEY not found.")
        else:
            model = genai.GenerativeModel("gemini-2.5-flash")

            prompt = f"""
            Create an executive summary for this dataset.

            Dataset Shape:
            {df.shape}

            Columns:
            {list(df.columns)}
            """

            response = model.generate_content(prompt)

            st.success("Executive Summary")
            st.write(response.text)        

    st.subheader("📌 Executive Summary")

    if numeric_cols:

        summary_col = numeric_cols[0]

        st.info(
            f"""
            Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.

            Average {summary_col}: {df[summary_col].mean():,.2f}

            Maximum {summary_col}: {df[summary_col].max():,.2f}

            Missing Values: {df.isnull().sum().sum()}
            """
        )
    
    # -----------------------------
    # Anomaly Detection
    # -----------------------------
    st.subheader("🚨 Anomaly Detection")

    if numeric_cols:

        anomaly_col = st.selectbox(
            "Select Column for Anomaly Detection",
            numeric_cols,
            key="anomaly"
        )

        anomaly_df = df[[anomaly_col]].dropna().copy()

        model = IsolationForest(
            contamination=0.05,
            random_state=42
        )

        anomaly_df["Anomaly"] = model.fit_predict(
            anomaly_df[[anomaly_col]]
        )

        fig_anomaly = px.scatter(
            anomaly_df,
            y=anomaly_col,
            color=anomaly_df["Anomaly"].astype(str),
            title=f"Anomaly Detection - {anomaly_col}"
        )

        st.plotly_chart(
            fig_anomaly,
            use_container_width=True
        )

    def create_pdf_report(text):
        buffer = BytesIO()

        p = canvas.Canvas(buffer)

        p.setFont("Helvetica", 12)

        y = 800

        for line in text.split("\n"):

            p.drawString(50, y, line[:100])

            y -= 20

            if y < 50:
                p.showPage()
                y = 800

        p.save()

        buffer.seek(0)

        return buffer                
    # -----------------------------
    # AI Business Insights
    # -----------------------------
    st.subheader("🤖 AI Business Insights")

    if st.button("Generate AI Insights"):

        if not api_key:
            st.error("GEMINI_API_KEY not found in .env file.")
        else:
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")

                sample_data = df.sample(
                    min(len(df), 50),
                    random_state=42
                ).to_csv(index=False)

                prompt = f"""
            You are an expert business analyst.

            Analyze the following dataset sample:

            {sample_data}

            Provide:
            1. Top 5 key insights
            2. Business risks
            3. Actionable recommendations

            Keep the response concise and professional.
            """

                with st.spinner("Thinking..."):
                    response = model.generate_content(prompt)

                st.success("Answer")
                st.write(response.text)

                pdf_file = create_pdf_report(response.text)
                
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_file,
                    file_name="AI_Report.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error: {e}")
    st.subheader("💬 Chat with Your Data")

    question = st.text_input("Ask a question about your data")

    if st.button("Ask AI"):

        if not api_key:
            st.error("GEMINI_API_KEY not found.")

        elif question.strip() == "":
            st.warning("Please enter a question.")

        else:
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")

                sample_data = df.sample(
                    min(len(df), 50),
                    random_state=42
                ).to_csv(index=False)

                prompt = f"""
    You are a professional data analyst.

    Dataset:
    {sample_data}

    Question:
    {question}

    Rules:
     - Answer based on the dataset.
     - If information is unavailable, clearly say so.
     - Give concise business-friendly insights.
     - Mention important trends when relevant.
    """
                with st.spinner("Thinking..."):
                    response = model.generate_content(prompt)

                st.success("Answer")
                st.write(response.text)

            except Exception as e:
                st.error(f"Error: {e}")
       
else:
    st.info("Please upload a CSV or Excel file.")
    
st.markdown("---")
st.markdown(
    "🚀 Built by Adil Maniyar | InsightPilot AI"
)    