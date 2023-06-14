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
openai.api_key = 'open ai key'

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


def generate_questions(summary, difficulty):
    """
    This function takes a summary and difficulty level as input and generates 5 questions using the OpenAI model.
    """

    # Initialize the OpenAI chat model
    # Adjust the temperature based on the difficulty level
    # print(difficulty)
    if difficulty == 1:
        temperature = 0.1
        prompt = f"I have read a text about the following topic: {summary}. Please generate 5 very easy questions based on the content. make the questions as easy as possible."

    elif difficulty == 2:
        temperature = 0.2
        prompt = f"I have read a text about the following topic: {summary}. Please generate 5 easy questions based on the content. make the questions with a below average difficulty."

    elif difficulty == 3:
        temperature = 0.3
        prompt = f"I have read a text about the following topic: {summary}. Please generate 5 average questions based on the content. make the questions as average as possible."

    elif difficulty == 4:
        temperature = 0.4
        prompt = f"I have read a text about the following topic: {summary}. Please generate 5 pretty hard questions based on the content. make the questions pretty hard to understand, but not too much."

    elif difficulty == 5:
        temperature = 0.5
        prompt = f"I have read a text about the following topic: {summary}. Please generate 5 very hard questions based on the content. make the questions as hard as possible."

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)

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


def summarize_text(db, summary_detail):
    # Use the logic of question answering to ask the model to summarize the text

    # Determine max_length based on the summary_detail
    if summary_detail == 1:
        summary_question = "Can you give me a simple, easy-to-understand summary of this text? Just cover the main points briefly."
    elif summary_detail == 2:
        summary_question = "Can you provide a summary of this text that includes important details but remains fairly concise?"
    elif summary_detail == 3:
        summary_question = "Can you summarize this text for me, covering all key points and some finer details? Aim for a balance between brevity and comprehensiveness."
    elif summary_detail == 4:
        summary_question = "Can you provide a detailed summary of this text, digging deeper into the main points and also touching on the less crucial aspects?"
    elif summary_detail == 5:
        summary_question = "Can you provide a very in-depth summary of this text, including as many details as possible? Leave nothing important out."

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


def get_response_from_query(db, query, summary_needed, k=4):
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

        with st.expander('Summarise PDF'):
            st.session_state.summary_detail = st.slider(
                'Set the detail level of your summary', 1, 5, 3)
            if 'summary' in st.session_state:
                summary = st.session_state.summary
                st.write(summary)
            elif knowledge_base:
                summary = summarize_text(
                    knowledge_base, st.session_state.summary_detail)
                st.session_state.summary = summary
                st.write(summary)
                db = FAISS.from_texts([summary], embeddings)
            else:
                st.write("No summary available")

        st.session_state.difficulty = st.slider(
            'Set the difficulty level of your quiz questions', 1, 5)

        # Check for changes in difficulty level or the Generate Questions button being pressed
        if 'last_difficulty' in st.session_state and st.session_state.difficulty != st.session_state.last_difficulty:
            st.session_state.last_difficulty = st.session_state.difficulty
            questions = generate_questions(
                st.session_state.summary, st.session_state.difficulty)
            st.session_state.questions = questions
        elif st.button("Generate Questions"):
            questions = generate_questions(
                st.session_state.summary, st.session_state.difficulty)
            st.session_state.questions = questions

        if 'answers' not in st.session_state:
            st.session_state.answers = {}

        if 'questions' not in st.session_state:
            st.session_state.questions = []

        if st.session_state.questions:  # Check if questions exist
            st.write('Generated Questions')
            for idx, question in enumerate(st.session_state.questions, start=1):
                st.write(f"Q{idx}: {question}")
                user_answer = st.text_input(f"Your Answer for Q{idx}")
                st.session_state.answers[idx] = {
                    'question': question, 'answer': user_answer}

            score = 0

            if st.button('Check All Answers'):
                for idx, data in st.session_state.answers.items():
                    question = data['question']
                    user_answer = data['answer']
                    is_correct, correct_answer = check_answer(
                        question, user_answer, st.session_state.summary, embeddings)
                    if is_correct:
                        score += 1
                        st.success(f'Your answer for Q{idx} is correct!')
                    else:
                        st.error(f'Your answer for Q{idx} is incorrect.')
                        with st.expander(f'Click to see the correct answer for question {idx}.'):
                            st.write("Question: ", question)
                            st.write("Correct Answer: ", correct_answer)

                st.write(
                    f"Your score: {score} out of {len(st.session_state.questions)}")
                progress_bar = st.progress(0)
                progress_percentage = score / len(st.session_state.questions)
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
