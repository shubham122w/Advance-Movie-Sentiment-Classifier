import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="🎬 Advanced Movie Review Sentiment Classifier",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🎬 Advanced Movie Review Sentiment Classifier</h1>', unsafe_allow_html=True)

# Initialize session state for text area
if 'review_text' not in st.session_state:
    st.session_state.review_text = ""

if 'batch_reviews_text' not in st.session_state:
    st.session_state.batch_reviews_text = ""

# Sidebar for additional features
with st.sidebar:
    st.header("⚙ Settings")

    # Model selection
    model_option = st.selectbox(
        "Choose Sentiment Model:",
        ["DistilBERT (Default)", "Custom Threshold", "Fine-tuned Model"]
    )

    # Confidence threshold adjustment
    confidence_threshold = st.slider(
        "Confidence Threshold:",
        min_value=0.5,
        max_value=0.95,
        value=0.6,
        step=0.05,
        help="Adjust the minimum confidence required for Positive/Negative classification"
    )

    # Theme selection
    theme = st.radio("Choose Theme:", ["Light", "Dark", "Colorful"])

    st.header("📊 History")
    show_history = st.checkbox("Show Analysis History", value=True)

    st.header("ℹ About")
    st.info(
        "This app uses Hugging Face's transformers to analyze movie review sentiment with advanced visualization features.")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Your Movie Review")

    # Text area with character counter - now using session state
    review = st.text_area(
        "Write your review here:",
        height=150,
        placeholder="This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout...",
        value=st.session_state.review_text,
        key="main_review_area"
    )

    # Update session state when user types
    st.session_state.review_text = review

    # Character counter
    if review:
        st.caption(f"Characters: {len(review)}/1000")

    # Example reviews
    with st.expander("💡 Need inspiration? Click for example reviews"):
        example1 = "This movie was a masterpiece! Brilliant acting, stunning visuals, and an emotional storyline that kept me captivated from start to finish."
        example2 = "Terrible movie. Poor acting, boring plot, and wasted my time. Would not recommend to anyone."
        example3 = "It was okay. Some parts were interesting but overall nothing special. Average movie experience."

        col_ex1, col_ex2, col_ex3 = st.columns(3)

        with col_ex1:
            if st.button("Positive Example", key="pos_ex"):
                st.session_state.review_text = example1
                st.rerun()

        with col_ex2:
            if st.button("Negative Example", key="neg_ex"):
                st.session_state.review_text = example2
                st.rerun()

        with col_ex3:
            if st.button("Neutral Example", key="neu_ex"):
                st.session_state.review_text = example3
                st.rerun()

    # Button container
    col1_1, col1_2, col1_3 = st.columns(3)

    with col1_1:
        classify_btn = st.button("🎯 Classify Sentiment", type="primary", use_container_width=True)

    with col1_2:
        clear_btn = st.button("🗑 Clear Text", use_container_width=True)

    with col1_3:
        analyze_btn = st.button("📈 Detailed Analysis", use_container_width=True)

with col2:
    st.subheader("📋 Quick Features")

    # Batch analysis
    st.write("*Multiple Reviews*")
    batch_reviews = st.text_area(
        "Enter multiple reviews (one per line):",
        height=100,
        placeholder="Review 1\nReview 2\nReview 3",
        value=st.session_state.batch_reviews_text,
        key="batch_review_area"
    )

    # Update batch reviews session state
    st.session_state.batch_reviews_text = batch_reviews

    batch_analyze_btn = st.button("Analyze Batch", use_container_width=True)

    # Clear batch button
    if st.button("Clear Batch", use_container_width=True):
        st.session_state.batch_reviews_text = ""
        st.rerun()

    # Language detection warning
    st.warning("⚠ Note: This model works best with English text.")

# Initialize session state for history
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []


# Load model with caching
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", return_all_scores=True)


classifier = load_model()

# Handle clear button action
if clear_btn:
    st.session_state.review_text = ""
    st.rerun()

# Handle classification
if classify_btn or analyze_btn:
    review_text = st.session_state.review_text
    if review_text.strip() == "":
        st.warning("Please enter a review to classify.")
    else:
        # Show loading spinner
        with st.spinner("Analyzing sentiment... This may take a few seconds."):
            # Simulate processing time for better UX
            time.sleep(0.5)

            # Get sentiment prediction
            results = classifier(review_text)
            positive_score = [score for score in results[0] if score['label'] == 'POSITIVE'][0]['score']
            negative_score = [score for score in results[0] if score['label'] == 'NEGATIVE'][0]['score']

            # Determine sentiment based on scores and threshold
            if positive_score > negative_score and positive_score > confidence_threshold:
                sentiment = "Positive 😀"
                sentiment_class = "positive"
                confidence = positive_score
            elif negative_score > positive_score and negative_score > confidence_threshold:
                sentiment = "Negative 😞"
                sentiment_class = "negative"
                confidence = negative_score
            else:
                sentiment = "Neutral 😐"
                sentiment_class = "neutral"
                confidence = max(positive_score, negative_score)

            # Store in history
            history_entry = {
                'review': review_text[:100] + "..." if len(review_text) > 100 else review_text,
                'sentiment': sentiment,
                'confidence': confidence,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.analysis_history.append(history_entry)

        # Display results with colored boxes
        st.markdown(f'<div class="sentiment-{sentiment_class}">', unsafe_allow_html=True)
        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"*Confidence Score:* {confidence:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Detailed analysis section
        if analyze_btn:
            st.subheader("📊 Detailed Analysis")

            # Create metrics columns
            col_met1, col_met2, col_met3 = st.columns(3)

            with col_met1:
                st.metric("Positive Score", f"{positive_score:.2%}")

            with col_met2:
                st.metric("Negative Score", f"{negative_score:.2%}")

            with col_met3:
                st.metric("Confidence Level",
                          "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low")

            # Create visualization
            fig = go.Figure()

            fig.add_trace(go.Bar(
                y=['Positive', 'Negative'],
                x=[positive_score, negative_score],
                orientation='h',
                marker_color=['#28a745', '#dc3545'],
                text=[f'{positive_score:.2%}', f'{negative_score:.2%}'],
                textposition='auto',
            ))

            fig.update_layout(
                title="Sentiment Distribution",
                xaxis_title="Confidence Score",
                yaxis_title="Sentiment",
                height=300,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Review statistics
            st.subheader("📝 Review Statistics")
            col_stat1, col_stat2, col_stat3 = st.columns(3)

            with col_stat1:
                st.metric("Word Count", len(review_text.split()))

            with col_stat2:
                st.metric("Character Count", len(review_text))

            with col_stat3:
                reading_time = max(1, len(review_text.split()) // 200)
                st.metric("Est. Reading Time", f"{reading_time} min")

# Batch analysis results
if batch_analyze_btn and st.session_state.batch_reviews_text.strip():
    st.subheader("📦 Batch Analysis Results")

    batch_results = []
    reviews_list = [r.strip() for r in st.session_state.batch_reviews_text.split('\n') if r.strip()]

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, rev in enumerate(reviews_list):
        if rev.strip():
            status_text.text(f"Analyzing review {i + 1} of {len(reviews_list)}...")
            result = classifier(rev)[0]
            pos_score = [score for score in result if score['label'] == 'POSITIVE'][0]['score']
            neg_score = [score for score in result if score['label'] == 'NEGATIVE'][0]['score']

            if pos_score > neg_score and pos_score > confidence_threshold:
                batch_sentiment = "Positive"
            elif neg_score > pos_score and neg_score > confidence_threshold:
                batch_sentiment = "Negative"
            else:
                batch_sentiment = "Neutral"

            batch_results.append({
                'Review #': i + 1,
                'Preview': rev[:50] + "..." if len(rev) > 50 else rev,
                'Sentiment': batch_sentiment,
                'Positive Score': f"{pos_score:.2%}",
                'Negative Score': f"{neg_score:.2%}"
            })

            progress_bar.progress((i + 1) / len(reviews_list))

    status_text.text("Analysis complete!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()

    if batch_results:
        df = pd.DataFrame(batch_results)
        st.dataframe(df, use_container_width=True)

        # Batch summary
        sentiment_counts = df['Sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Batch Sentiment Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Display analysis history
if show_history and st.session_state.analysis_history:
    st.subheader("📜 Analysis History")

    history_df = pd.DataFrame(st.session_state.analysis_history[-10:])  # Show last 10 entries
    st.dataframe(history_df, use_container_width=True)

    # Clear history button
    if st.button("Clear History"):
        st.session_state.analysis_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤ using Streamlit and Hugging Face Transformers | "
    "Model: DistilBERT sentiment analysis"
)