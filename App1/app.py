# Core Pkgs
import streamlit as st
import altair as alt


# EDA Pkgs
import pandas as pd
import numpy as np

# Utils
import joblib
pipe_lr = joblib.load(open("Models1/cpc_classifier_pipe_lr_1_Mar_2022.pkl", "rb"))


# Functions to predict label
def predict_label(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


# Main application
def main():
    st.title('Patent Classification App')
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("CPC from 1st Main Claim")
        
        with st.form(key='cpc_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            # Apply function here
            prediction = predict_label(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success('Original Text')
                st.write(raw_text)

                st.success('Prediction')
                st.write(prediction)
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success('Prediction probability')
                # st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ['classes', 'probability']

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='classes', y='probability', color='classes')
                st.altair_chart(fig, use_container_width=True)

    elif choice == "Monitor":
        st.subheader("Monitor App")

    else:
        st.subheader("About")
        

if __name__ == '__main__':
    main()
