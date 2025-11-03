import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources (only the first time)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model (make sure these files exist and are fitted)

with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# Streamlit app


# st.markdown("""
#     <style>
#         body {
#             background-image: url('knit_img.jpg');
#             background-size: cover;
#             background-position: center;
#             background-attachment: fixed;
#             color: white;
#         }
#     </style>
# """, unsafe_allow_html=True)

st.set_page_config(page_title="Project Information", page_icon="📚")
st.title("📨 Email/SMS Spam Classifier")

st.subheader("Kamla Nehru Institute Of Technology,Sultanpur")

st.subheader("Mentor and Student Information")
# Project Mentor
mentor_name = "Prof. Samir Srivastava"  # Replace with actual mentor name
mentor_designation = "Associate Professor, Department of Computer Science, KNIT Sultanpur"
mentor_name2 = "Prof. Vinay Singh"  # Replace with actual mentor name
mentor_designation2 = "Associate Professor, Department of Computer Science,KNIT Sultanpur"

student_names = ["Dheeraj Singh", "Yogesh Pandey", "Narendra Singh"]


st.markdown(f"""
The project is supervised by **{mentor_name}**, a highly respected {mentor_designation}. Prof. {mentor_name2} has extensive experience in the field and has guided numerous students to successful project outcomes.

The project is being carried out by a team of dedicated students: **{', '.join(student_names)}**. We as students are working together to apply our skills and knowledge to the task, with a strong focus on delivering high-quality results. The collaboration among the team members and guidance from the mentor ensures that the project is on the right track for success.
""")


# Project Students
student_names = ["Dheeraj Singh", "Yogesh Pandey", "Narendra Singh"]

input_sms = st.text_area("Enter the message:")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        try:
            vector_input = tfidf.transform([transformed_sms])

            # 3. Predict
            result = model.predict(vector_input)[0]

            # 4. Display result
            if result == 1:
                st.header("🚨 Spam")
            else:
                st.header("✅ Not Spam")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
