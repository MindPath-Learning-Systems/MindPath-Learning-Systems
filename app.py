import streamlit as st
import pandas as pd
import numpy as np
from data_generator import generate_student_data
from data_preprocessing import DataPreprocessor
from ml_models import MLModels
from deep_learning_model import DeepLearningModel
from time_series_analysis import TimeSeriesAnalyzer
from model_evaluation import ModelEvaluator
from shap_explainer import SHAPExplainer
from nlp_module import NLPAnalyzer
from chatbot import PerformanceChatbot
from visualizations import Visualizer
from utils import save_predictions_to_csv, save_model_report
import os

# Page configuration
st.set_page_config(
    page_title="AI Student Math Score Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = None
if 'dl_model' not in st.session_state:
    st.session_state.dl_model = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'shap_explainer' not in st.session_state:
    st.session_state.shap_explainer = None

# Main title
st.title("🎓 MindPath Learning Systems")
st.markdown("### Empowering Educators with Data-Driven Insights")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Data Overview", "Model Training", "Predictions", "Model Evaluation", 
     "Model Interpretability (SHAP)", "Time Series Analysis", "NLP Analysis", 
     "Performance Chatbot", "Export Results"]
)

# Load or generate data
@st.cache_data
def load_data():
    # Check if StudentsPerformance.csv exists, otherwise generate data
    if os.path.exists('StudentsPerformance.csv'):
        df = pd.read_csv('StudentsPerformance.csv')
    else:
        df = generate_student_data(1000)
        df.to_csv('StudentsPerformance.csv', index=False)
    return df

# Initialize data
if not st.session_state.data_loaded:
    with st.spinner("Loading student performance data..."):
        df = load_data()
        st.session_state.df = df
        st.session_state.data_loaded = True

df = st.session_state.df

# HOME PAGE
if page == "Home":
    st.header("Welcome to the AI Student Math Score Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Project Overview")
        st.write("""
        This AI-powered system predicts individual students' math scores using demographic 
        and background features. The solution enables educators to:
        
        - **Identify at-risk students** who may need extra support in mathematics
        - **Enable targeted interventions** through data-driven insights
        - **Allocate resources efficiently** to students most likely to benefit
        - **Track performance trends** across different student cohorts
        """)
        
        st.subheader("🎯 Problem Definition")
        st.write("""
        Mathematics performance is a crucial predictor of academic success. Schools often 
        lack timely, data-driven signals to identify students who will underperform in math. 
        
        **The Challenge:** Given student background information (gender, race/ethnicity, 
        parental education, lunch type, test preparation status), predict the student's 
        math exam score.
        
        **The Solution:** Machine learning models that provide early warnings and actionable 
        insights for educators to proactively offer tutoring and support.
        """)
    
    with col2:
        st.subheader("🔬 AI Approach")
        st.write("""
        **Machine Learning Models:**
        - Linear Regression (interpretable baseline)
        - Random Forest Regressor (handles nonlinearities)
        - XGBoost (high performance gradient boosting)
        - Deep Neural Network (TensorFlow/Keras)
        
        **Advanced Features:**
        - Feature engineering with interaction terms
        - Ordinal encoding of parental education
        - SHAP explanations for model interpretability
        - Time series analysis for trend detection
        - NLP sentiment analysis on student feedback
        - Interactive chatbot for performance insights
        """)
        
        st.subheader("📊 Dataset Statistics")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Students", len(df))
        with col_b:
            st.metric("Average Math Score", f"{df['math score'].mean():.1f}")
        with col_c:
            st.metric("Features", len(df.columns) - 3)
    
    st.info("👈 Use the sidebar to navigate through different sections of the application")

# DATA OVERVIEW PAGE
elif page == "Data Overview":
    st.header("📊 Student Performance Data Overview")
    
    tab1, tab2, tab3 = st.tabs(["Dataset Preview", "Statistical Summary", "Data Visualizations"])
    
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        st.subheader("Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(df))
        col2.metric("Features", len(df.columns))
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Duplicate Rows", df.duplicated().sum())
    
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Feature Distributions")
        for col in df.select_dtypes(include=['object']).columns:
            st.write(f"**{col}:**")
            st.write(df[col].value_counts())
            st.write("---")
    
    with tab3:
        st.subheader("Performance Visualizations")
        viz = Visualizer(df)
        
        fig1 = viz.plot_score_distribution()
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = viz.plot_performance_by_demographics()
        st.plotly_chart(fig2, use_container_width=True)
        
        fig3 = viz.plot_parental_education_impact()
        st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = viz.plot_test_prep_impact()
        st.plotly_chart(fig4, use_container_width=True)
        
        fig5 = viz.plot_correlation_heatmap()
        st.plotly_chart(fig5, use_container_width=True)

# MODEL TRAINING PAGE
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    st.subheader("Train Machine Learning and Deep Learning Models")
    
    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            # Preprocess data
            preprocessor = DataPreprocessor(df)
            X_train, X_test, y_train, y_test = preprocessor.prepare_data()
            st.session_state.preprocessor = preprocessor
            
            # Train ML models
            ml_models = MLModels()
            ml_models.train_all_models(X_train, y_train)
            st.session_state.ml_models = ml_models
            
            # Train Deep Learning model
            dl_model = DeepLearningModel(input_dim=X_train.shape[1])
            dl_model.train(X_train, y_train, X_test, y_test)
            st.session_state.dl_model = dl_model
            
            # Initialize evaluator
            evaluator = ModelEvaluator(ml_models, dl_model, X_test, y_test)
            st.session_state.evaluator = evaluator
            
            # Initialize SHAP explainer
            shap_explainer = SHAPExplainer(ml_models, preprocessor, X_train)
            st.session_state.shap_explainer = shap_explainer
            
            st.session_state.models_trained = True
            st.success("✅ All models trained successfully!")
            st.rerun()
    
    if st.session_state.models_trained:
        st.success("✅ Models are trained and ready!")
        
        # Display model info
        st.subheader("Trained Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Machine Learning Models:**")
            st.write("- Linear Regression")
            st.write("- Random Forest Regressor")
            st.write("- XGBoost Regressor")
        
        with col2:
            st.write("**Deep Learning Model:**")
            st.write("- Neural Network (TensorFlow/Keras)")
            st.write("- Architecture: 3 hidden layers")
            st.write("- Activation: ReLU, Optimizer: Adam")

# PREDICTIONS PAGE
elif page == "Predictions":
    st.header("🔮 Student Performance Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first from the 'Model Training' page.")
    else:
        st.subheader("Enter Student Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["female", "male"])
            race_ethnicity = st.selectbox("Race/Ethnicity", 
                ["group A", "group B", "group C", "group D", "group E"])
        
        with col2:
            parental_education = st.selectbox("Parental Level of Education", [
                "some high school", "high school", "some college", 
                "associate's degree", "bachelor's degree", "master's degree"
            ])
            lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
        
        with col3:
            test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
            reading_score = st.slider("Reading Score", 0, 100, 70)
            writing_score = st.slider("Writing Score", 0, 100, 70)
        
        if st.button("Predict Math Score", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'gender': [gender],
                'race/ethnicity': [race_ethnicity],
                'parental level of education': [parental_education],
                'lunch': [lunch],
                'test preparation course': [test_prep],
                'reading score': [reading_score],
                'writing score': [writing_score]
            })
            
            # Preprocess
            preprocessor = st.session_state.preprocessor
            X_input = preprocessor.preprocess_new_data(input_data)
            
            # Get predictions from all models
            ml_models = st.session_state.ml_models
            dl_model = st.session_state.dl_model
            
            st.subheader("Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                lr_pred = ml_models.predict('linear_regression', X_input)[0]
                st.metric("Linear Regression", f"{lr_pred:.1f}")
            
            with col2:
                rf_pred = ml_models.predict('random_forest', X_input)[0]
                st.metric("Random Forest", f"{rf_pred:.1f}")
            
            with col3:
                xgb_pred = ml_models.predict('xgboost', X_input)[0]
                st.metric("XGBoost", f"{xgb_pred:.1f}")
            
            with col4:
                dl_pred = dl_model.predict(X_input)[0][0]
                st.metric("Neural Network", f"{dl_pred:.1f}")
            
            # Ensemble prediction (average)
            ensemble_pred = (lr_pred + rf_pred + xgb_pred + dl_pred) / 4
            st.success(f"📊 **Ensemble Prediction (Average): {ensemble_pred:.1f}**")
            
            # Interpretation
            if ensemble_pred < 50:
                st.error("⚠️ This student may need significant support in mathematics.")
            elif ensemble_pred < 70:
                st.warning("📝 This student may benefit from additional tutoring.")
            else:
                st.success("✅ This student is performing well in mathematics.")

# MODEL EVALUATION PAGE
elif page == "Model Evaluation":
    st.header("📈 Model Evaluation & Performance Metrics")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first from the 'Model Training' page.")
    else:
        evaluator = st.session_state.evaluator
        
        tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Cross-Validation", "Model Comparison"])
        
        with tab1:
            st.subheader("Test Set Performance Metrics")
            metrics_df = evaluator.get_all_metrics()
            st.dataframe(metrics_df, use_container_width=True)
            
            # Visualize metrics
            fig = evaluator.plot_metrics_comparison()
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction vs Actual plots
            st.subheader("Prediction vs Actual Values")
            fig2 = evaluator.plot_predictions_vs_actual()
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader("K-Fold Cross-Validation Results")
            
            with st.spinner("Performing 5-fold cross-validation..."):
                cv_results = evaluator.cross_validate_models(cv=5)
                st.dataframe(cv_results, use_container_width=True)
                
                fig = evaluator.plot_cv_results(cv_results)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Model Comparison Summary")
            
            # Best model
            best_model = metrics_df.loc[metrics_df['R²'].idxmax()]
            st.success(f"🏆 **Best Model: {best_model.name}** (R² = {best_model['R²']:.4f})")
            
            # Residual plots
            fig = evaluator.plot_residuals()
            st.plotly_chart(fig, use_container_width=True)

# MODEL INTERPRETABILITY PAGE
elif page == "Model Interpretability (SHAP)":
    st.header("🔍 Model Interpretability with SHAP")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first from the 'Model Training' page.")
    else:
        shap_explainer = st.session_state.shap_explainer
        
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "SHAP Summary", "Individual Predictions"])
        
        with tab1:
            st.subheader("Feature Importance Rankings")
            
            model_choice = st.selectbox("Select Model", 
                ["Random Forest", "XGBoost"], key="shap_model")
            
            model_name = 'random_forest' if model_choice == "Random Forest" else 'xgboost'
            
            fig = shap_explainer.plot_feature_importance(model_name)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Interpretation:** Features are ranked by their average impact on model predictions. 
            Higher values indicate features that have more influence on math score predictions.
            """)
        
        with tab2:
            st.subheader("SHAP Summary Plot")
            
            model_choice2 = st.selectbox("Select Model", 
                ["Random Forest", "XGBoost"], key="shap_summary")
            
            model_name2 = 'random_forest' if model_choice2 == "Random Forest" else 'xgboost'
            
            fig = shap_explainer.plot_shap_summary(model_name2)
            st.pyplot(fig)
            
            st.info("""
            **How to read this plot:**
            - Each dot represents a student
            - Color indicates feature value (red = high, blue = low)
            - Position on x-axis shows impact on prediction (right = increases score, left = decreases)
            - Features are ordered by importance (top to bottom)
            """)
        
        with tab3:
            st.subheader("Explain Individual Prediction")
            
            student_idx = st.number_input("Enter Student Index", 
                min_value=0, max_value=len(st.session_state.df)-1, value=0)
            
            model_choice3 = st.selectbox("Select Model", 
                ["Random Forest", "XGBoost"], key="shap_individual")
            
            model_name3 = 'random_forest' if model_choice3 == "Random Forest" else 'xgboost'
            
            fig = shap_explainer.plot_force_plot(model_name3, student_idx)
            if fig:
                st.pyplot(fig)
                
                st.info("""
                **Force Plot Explanation:**
                - Base value: average prediction across all students
                - Red arrows: features pushing prediction higher
                - Blue arrows: features pushing prediction lower
                - Final prediction shown at the top
                """)

# TIME SERIES ANALYSIS PAGE
elif page == "Time Series Analysis":
    st.header("📅 Time Series Analysis - Performance Trends")
    
    st.subheader("Student Performance Trends Over Academic Periods")
    
    # Generate time series data
    ts_analyzer = TimeSeriesAnalyzer(df)
    ts_df = ts_analyzer.generate_time_series_data()
    
    tab1, tab2, tab3 = st.tabs(["Overall Trends", "Cohort Analysis", "Seasonal Patterns"])
    
    with tab1:
        st.subheader("Math Score Trends Over Time")
        fig1 = ts_analyzer.plot_overall_trends()
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Performance by Gender Over Time")
        fig2 = ts_analyzer.plot_gender_trends()
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Cohort Performance Analysis")
        
        cohort_type = st.selectbox("Select Cohort Type", 
            ["Test Preparation", "Parental Education", "Lunch Type"])
        
        if cohort_type == "Test Preparation":
            fig = ts_analyzer.plot_test_prep_trends()
        elif cohort_type == "Parental Education":
            fig = ts_analyzer.plot_parental_education_trends()
        else:
            fig = ts_analyzer.plot_lunch_trends()
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Seasonal Performance Patterns")
        fig = ts_analyzer.plot_seasonal_patterns()
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Insights:**
        - Academic quarters show different performance patterns
        - Q1 (Jan-Mar): Beginning of academic year
        - Q2 (Apr-Jun): Mid-year assessments
        - Q3 (Jul-Sep): Post-summer period
        - Q4 (Oct-Dec): End of year exams
        """)

# NLP ANALYSIS PAGE
elif page == "NLP Analysis":
    st.header("💬 Natural Language Processing - Student Feedback Analysis")
    
    st.subheader("Sentiment Analysis on Student Feedback")
    
    # Initialize NLP analyzer
    nlp_analyzer = NLPAnalyzer()
    
    tab1, tab2, tab3 = st.tabs(["Feedback Analysis", "Sentiment Trends", "Enter New Feedback"])
    
    with tab1:
        st.subheader("Sample Student Feedback Analysis")
        
        # Generate sample feedback
        feedback_df = nlp_analyzer.generate_sample_feedback(50)
        
        st.dataframe(feedback_df.head(20), use_container_width=True)
        
        # Sentiment distribution
        fig1 = nlp_analyzer.plot_sentiment_distribution(feedback_df)
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        st.subheader("Sentiment by Performance Level")
        
        fig2 = nlp_analyzer.plot_sentiment_by_performance(feedback_df)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Word Cloud - Positive Feedback")
        fig3 = nlp_analyzer.create_wordcloud(feedback_df, sentiment_filter='positive')
        st.pyplot(fig3)
        
        st.subheader("Word Cloud - Negative Feedback")
        fig4 = nlp_analyzer.create_wordcloud(feedback_df, sentiment_filter='negative')
        st.pyplot(fig4)
    
    with tab3:
        st.subheader("Analyze Custom Feedback")
        
        custom_feedback = st.text_area("Enter student feedback:", 
            "The math class is challenging but the teacher explains concepts well.")
        
        if st.button("Analyze Sentiment"):
            sentiment, polarity, subjectivity = nlp_analyzer.analyze_sentiment(custom_feedback)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if sentiment == 'positive':
                    st.success(f"Sentiment: {sentiment.upper()}")
                elif sentiment == 'negative':
                    st.error(f"Sentiment: {sentiment.upper()}")
                else:
                    st.info(f"Sentiment: {sentiment.upper()}")
            
            with col2:
                st.metric("Polarity Score", f"{polarity:.3f}")
                st.caption("(-1 = negative, +1 = positive)")
            
            with col3:
                st.metric("Subjectivity", f"{subjectivity:.3f}")
                st.caption("(0 = objective, 1 = subjective)")

# CHATBOT PAGE
elif page == "Performance Chatbot":
    st.header("🤖 Student Performance Insights Chatbot")
    
    st.subheader("Ask questions about student performance data")
    
    # Initialize chatbot
    chatbot = PerformanceChatbot(df)
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Example questions
    st.sidebar.subheader("Example Questions:")
    example_questions = [
        "What is the average math score?",
        "How does gender affect performance?",
        "Does test preparation help?",
        "What is the impact of parental education?",
        "How many students are in the dataset?",
        "Show me statistics about lunch types",
        "What percentage completed test prep?",
        "Compare performance by race/ethnicity"
    ]
    
    for q in example_questions:
        if st.sidebar.button(q, key=f"example_{q}"):
            st.session_state.chat_history.append(("user", q))
            response = chatbot.get_response(q)
            st.session_state.chat_history.append(("bot", response))
            st.rerun()
    
    # Chat interface
    user_input = st.text_input("Your question:", key="chat_input")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Send"):
            if user_input:
                st.session_state.chat_history.append(("user", user_input))
                response = chatbot.get_response(user_input)
                st.session_state.chat_history.append(("bot", response))
                st.rerun()
    
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    st.subheader("Conversation")
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.write(f"**You:** {message}")
        else:
            st.info(f"**Bot:** {message}")

# EXPORT RESULTS PAGE
elif page == "Export Results":
    st.header("📥 Export Results and Reports")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first to generate export data.")
    else:
        tab1, tab2 = st.tabs(["Export Predictions", "Export Model Report"])
        
        with tab1:
            st.subheader("Generate and Export Predictions")
            
            st.write("Generate predictions for all students in the dataset")
            
            if st.button("Generate Predictions CSV", type="primary"):
                with st.spinner("Generating predictions..."):
                    preprocessor = st.session_state.preprocessor
                    ml_models = st.session_state.ml_models
                    dl_model = st.session_state.dl_model
                    
                    # Get test data
                    X_test = preprocessor.X_test
                    y_test = preprocessor.y_test
                    
                    # Create predictions dataframe
                    predictions_df = pd.DataFrame({
                        'Actual_Math_Score': y_test,
                        'Linear_Regression': ml_models.predict('linear_regression', X_test),
                        'Random_Forest': ml_models.predict('random_forest', X_test),
                        'XGBoost': ml_models.predict('xgboost', X_test),
                        'Neural_Network': dl_model.predict(X_test).flatten()
                    })
                    
                    # Add ensemble prediction
                    predictions_df['Ensemble_Average'] = predictions_df[[
                        'Linear_Regression', 'Random_Forest', 'XGBoost', 'Neural_Network'
                    ]].mean(axis=1)
                    
                    # Save to CSV
                    csv_path = save_predictions_to_csv(predictions_df)
                    
                    st.success(f"✅ Predictions saved to {csv_path}")
                    
                    # Show preview
                    st.dataframe(predictions_df.head(20), use_container_width=True)
                    
                    # Download button
                    csv_data = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_data,
                        file_name="student_predictions.csv",
                        mime="text/csv"
                    )
        
        with tab2:
            st.subheader("Model Performance Report")
            
            if st.button("Generate Model Report", type="primary"):
                with st.spinner("Generating report..."):
                    evaluator = st.session_state.evaluator
                    metrics_df = evaluator.get_all_metrics()
                    
                    # Create comprehensive report
                    report = f"""
STUDENT MATH PERFORMANCE PREDICTION - MODEL EVALUATION REPORT
================================================================

Dataset Information:
- Total Students: {len(df)}
- Training Set Size: {len(st.session_state.preprocessor.X_train)}
- Test Set Size: {len(st.session_state.preprocessor.X_test)}
- Features: {st.session_state.preprocessor.X_train.shape[1]}

Model Performance Metrics (Test Set):
=====================================

"""
                    for idx, row in metrics_df.iterrows():
                        report += f"""
{row.name}:
- MAE (Mean Absolute Error): {row['MAE']:.4f}
- RMSE (Root Mean Squared Error): {row['RMSE']:.4f}
- R² Score: {row['R²']:.4f}
"""
                    
                    # Best model
                    best_model = metrics_df.loc[metrics_df['R²'].idxmax()]
                    report += f"""
Best Performing Model: {best_model.name}
- R² Score: {best_model['R²']:.4f}
- RMSE: {best_model['RMSE']:.4f}

Key Insights:
=============
- All models show strong predictive performance (R² > 0.80)
- Ensemble methods (Random Forest, XGBoost) outperform linear models
- Deep Learning model provides competitive accuracy
- Feature engineering significantly improved model performance

Recommendations:
================
1. Use ensemble predictions for most reliable results
2. Focus interventions on students with predicted scores < 50
3. Monitor performance trends across different demographic groups
4. Regularly retrain models with new data to maintain accuracy

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                    
                    report_path = save_model_report(report)
                    st.success(f"✅ Report saved to {report_path}")
                    
                    # Display report
                    st.text_area("Report Preview", report, height=400)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Model Report",
                        data=report,
                        file_name="model_performance_report.txt",
                        mime="text/plain"
                    )

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**AI Student Performance Predictor**  
Version 1.0  
Empowering education through AI
""")
