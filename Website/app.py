from dotenv import load_dotenv
import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import scipy.spatial

load_dotenv(find_dotenv())
openai.api_key = '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'
# load_dotenv()

embeddings = OpenAIEmbeddings()


def check_answer(question, user_answer, summary, embeddings):
    """
    Uses OpenAI to generate an answer to the selected question based on the summary.
    Compares the user's answer with the generated answer to determine if they convey the same meaning.
    """

    # Return False if the answer is empty, but also provide the correct answer
    if not user_answer.strip():
        # Generate answer from OpenAI model
        db = FAISS.from_texts([summary], embeddings)
        model_answer, _ = get_response_from_query(
            db, question, summary_needed=False)
        return False, model_answer

    # Generate answer from OpenAI model
    db = FAISS.from_texts([summary], embeddings)
    model_answer, _ = get_response_from_query(
        db, question, summary_needed=False)

    # Use SentenceTransformer to get sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    user_answer_emb = model.encode([user_answer])[0]
    model_answer_emb = model.encode([model_answer])[0]

    # Calculate cosine similarity
    cosine_sim = 1 - \
        scipy.spatial.distance.cosine(user_answer_emb, model_answer_emb)

    # If the cosine similarity is above a threshold, consider it as correct
    if cosine_sim > 0.7:
        return True, ""
    else:
        return False, model_answer


# def check_answer(question, user_answer, model_answer):
#     # Return False if the user's answer is empty
#     if not user_answer.strip():
#         return False

#     # Create a conversation with the model
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": f"The question is: {question}"},
#         {"role": "user", "content": f"The model's answer is: {model_answer}"},
#         {"role": "user", "content": f"The user's answer is: {user_answer}"},
#         {"role": "user", "content": "If you were a teacher, would you accept the user's answer as a correct answer? Give a score from 1 to 10 to show how likely you are to accept it."}
#     ]

#     # Get the model's response
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo", messages=messages)

#     # Check if the model thinks the answers match
#     response_text = response['choices'][0]['message']['content'].strip()
#     try:
#         similarity_score = float(response_text)
#     except ValueError:
#         return False  # If the model's response isn't a number, treat it as incorrect

#     # If the model gives a score of 7 or higher, consider it correct
#     is_correct = similarity_score >= 7.0

#     return is_correct


def generate_questions(summary):
    """
    This function takes a summary as input and generates 5 questions using the OpenAI model.
    """

    # Initialize the OpenAI chat model
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Prompt for generating questions
    prompt = f"I have read a text about the following topic: {summary}. Please generate 5 questions based on the content."

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that can generate questions based on the provided text: {docs}
        
        Your task is to generate 5 meaningful questions that relate to the content of the text.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # We wrap the system message into a ChatPromptTemplate
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    # Generate questions
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    questions = chain.run(question=prompt, docs=summary)

    # Assume the model returns the questions separated by newlines, split them
    questions = questions.split('\n')

    return questions


def summarize_text(db):
    # Use the logic of question answering to ask the model to summarize the text
    summary_question = "Can you summarize this text for me? Make sure to be specific and explain the most important subjects in high detail. Use a nice layout so it's easier to understand, so use new lines etc. Please use emoji's to make it cleaner."
    summary, _ = get_response_from_query(
        db, summary_question, summary_needed=True)
    return summary


def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4, summary_needed=False):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    if summary_needed == True:
        human_template = "Answer the following question: {question}"
    else:
        human_template = "Answer the following question, please keep it short and straight to the point: {question}"

    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


def main():
    # Initialize variables
    knowledge_base = None
    db = None

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""

    st.set_page_config(page_title="Study Material Learner")
    st.image("https://assets.website-files.com/5e318ddf83dd66053d55c38a/602ba7efd5e8b761ed988e2a_FBF%20logo%20wide.png",
             caption='Study Material Learner', use_column_width=True)

    st.title("Study Material Learner")
    st.write("""
    This is a tool specifically designed to accelerate learning for students, making their study experience more interactive. 
    It offers a unique capability to extract information from resources such as PDF files and lectures, and then transform this information into quizzes. 
    These quizzes are automatically generated from the student's provided study materials. 
    This function not only enhances the student's understanding but also contributes to their intellectual growth, making them smarter.
    """)

    option = st.selectbox('Choose your option',
                          ('Select', 'PDF', 'YouTube Video'))

    progress = st.progress(0)
    knowledge_base = None  # Initialize knowledge_base

    if option == 'PDF':
        pdf = st.sidebar.file_uploader("Upload your PDF", type="pdf")

        if 'user_question' not in st.session_state:
            st.session_state.user_question = ""

        user_question = st.sidebar.text_input(
            "Ask a question about your PDF:", value=st.session_state.user_question)

        ask_button_clicked = st.sidebar.button('Ask')

        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            progress.progress(10)
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text()

            with st.expander("Click to see PDF file text"):
                st.write(text)

            progress.progress(20)

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )

            chunks = text_splitter.split_text(text)
            progress.progress(40)
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            progress.progress(60)

            if user_question and ask_button_clicked:
                st.session_state.conversation_history.append(
                    ('You', user_question))
                docs = knowledge_base.similarity_search(user_question)
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")

                with get_openai_callback() as cb:
                    response = chain.run(
                        input_documents=docs, question=user_question)

                progress.progress(80)

                try:
                    progress.progress(100)
                    st.session_state.conversation_history.append(
                        ('Bot', response))

                except Exception as e:
                    st.error(f"An error occurred: {e}")

                st.session_state.user_question = ""

    if knowledge_base:
        # Place the expander outside the if-statement
        with st.expander('Summarise PDF'):
            # Check if the summary is already stored in the session_state
            if 'summary' in st.session_state:
                # If the summary is already in the session_state, just use it
                summary = st.session_state.summary
                st.write(summary)
            elif knowledge_base:
                # Generate the summary
                summary = summarize_text(knowledge_base)
                # Store the summary in the session_state
                st.session_state.summary = summary
                # Display the summary in the expander
                st.write(summary)
                # Create the database from the summary
                db = FAISS.from_texts([summary], embeddings)
            else:
                st.write("No summary available")

        # Check if the questions are not already stored in the session_state
        if 'questions' not in st.session_state:
            # Generate questions based on the summary
            questions = generate_questions(summary)
            # Store the questions in the session_state
            st.session_state.questions = questions
        else:  # If the questions are already in the session_state, just use them
            questions = st.session_state.questions

        if 'answers' not in st.session_state:
            st.session_state.answers = {}

    # Inside the 'Generated Questions' expander:
    # with st.expander('Generated Questions'):
        st.write('Generated Questions')
        for idx, question in enumerate(questions, start=1):
            st.write(f"Q{idx}: {question}")
            user_answer = st.text_input(f"Your Answer for Q{idx}")

            # Store the user's answer in the session state
            st.session_state.answers[idx] = {
                'question': question, 'answer': user_answer}

        # Initialize the score variable
        score = 0

        if st.button('Check All Answers'):
            for idx, data in st.session_state.answers.items():
                question = data['question']
                user_answer = data['answer']

                # Use the check_answer function
                is_correct, correct_answer = check_answer(
                    question, user_answer, summary, embeddings)
                if is_correct:
                    score += 1
                    st.success(f'Your answer for Q{idx} is correct!')
                else:
                    st.error(f'Your answer for Q{idx} is incorrect.')
                    with st.expander(f'Click to see the correct answer for question {idx}.'):
                        st.write(correct_answer)

            # Display the score
            st.write(f"Your score: {score} out of {len(questions)}")

            # Display a progress bar for the score
            progress_bar = st.progress(0)
            progress_percentage = score / len(questions)
            progress_bar.progress(progress_percentage)

    db = None  # Initialize db

    if option == 'YouTube Video':
        youtube_url = st.sidebar.text_input("Enter YouTube video URL:")

        # If a URL is entered, embed the video and process it
        if youtube_url:
            # Process the YouTube video transcript as you did in your existing code
            db = create_db_from_youtube_video_url(youtube_url)

            # Ask for a summary of the YouTube video
            summary_question = "Can you summarize this video for me? Make sure to be specific and explain the most important subjects in high detail. Use a nice layout so it's easier to understand. Please use emoji's to make it cleaner. "
            with st.expander("Summary of YouTube video"):
                summary, _ = get_response_from_query(
                    db, summary_question, summary_needed=True)
                # st.session_state.conversation_history.append(('Bot', summary))
                st.write(summary)

            # Embed the YouTube video
            st.video(youtube_url)
            # Ask a question about the YouTube video
            youtube_question = st.sidebar.text_input(
                "Ask a question about the YouTube video:", value=st.session_state.user_question)

            ask_youtube_button_clicked = st.sidebar.button(
                'Ask YouTube Question')

            if youtube_question and ask_youtube_button_clicked:
                st.session_state.conversation_history.append(
                    ('You', youtube_question))

                response, docs = get_response_from_query(
                    db, youtube_question, summary_needed=False)
                st.session_state.conversation_history.append(('Bot', response))

                # Clear the text input box
                st.session_state.user_question = ""

    # Only display the conversation section if there are any questions in the conversation history

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if st.session_state.conversation_history:
        st.sidebar.markdown("## Conversation")
        for role, message in st.session_state.conversation_history:
            if role == 'Bot':
                st.sidebar.markdown(
                    f'<div style="color: #6c757d;">{role}: {message}</div>', unsafe_allow_html=True)
            else:
                st.sidebar.markdown(f"{role}: {message}")


if __name__ == '__main__':
    main()
