from dotenv import load_dotenv
import streamlit as st
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
import textwrap
import os

load_dotenv(find_dotenv())
# openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()


# def get_questions_from_summary(summary, num_questions=5):
#     # Ask GPT-3 to generate questions about the summary
#     question_prompt = f"Generate {num_questions} questions about the following summary: {summary}"
#     questions, _ = get_response_from_query(db, question_prompt)
#     return questions.split("\n")  # Assume each question is on a new line


def summarize_text(db):
    # Use the logic of question answering to ask the model to summarize the text
    summary_question = "Can you summarize this text for me? Make sure to be specific and explain the most important subjects in high detail. Use a nice layout so it's easier to understand, so use new lines etc. Please use emoji's to make it cleaner."
    summary, _ = get_response_from_query(db, summary_question)
    return summary


def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
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
    human_template = "Answer the following question: {question}"
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
    #load_dotenv()
    # Initialize variables
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""
        
    knowledge_base = None
    db = None


    st.set_page_config(page_title="Study Material Learner")
    st.image("https://assets.website-files.com/5e318ddf83dd66053d55c38a/602ba7efd5e8b761ed988e2a_FBF%20logo%20wide.png",
             caption='Study Material Learner', use_column_width=True)

    st.title("Ask Your PDF or YouTube Video")
    st.write("""
    This app allows you to upload a PDF or input a YouTube video URL and ask questions about its content. 
    After you upload a PDF or input a YouTube URL, it will be processed and you will be able to ask any question about the content.
    """)

    # if 'conversation_history' not in st.session_state:
    #     st.session_state.conversation_history = []

    # if 'user_question' not in st.session_state:
    #     st.session_state.user_question = ""

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
                    print(cb)
                    
                progress.progress(80)

                try:
                    progress.progress(100)
                    st.session_state.conversation_history.append(
                        ('Bot', response))
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

                st.session_state.user_question = ""

        if knowledge_base:
            with st.expander('Summarise PDF'):
                summary = summarize_text(knowledge_base)
                st.write(summary)

            db = FAISS.from_texts([summary], embeddings)

            # Generate questions from the summary
            # with st.expander("Quiz questions"):
            #     if db is not None:
            #         questions = get_questions_from_summary(db, summary)
            #         for i, question in enumerate(questions, start=1):
            #             st.markdown(f"**Question {i}:** {question}")

    db = None  # Initialize db

    if option == 'YouTube Video':
        youtube_url = st.sidebar.text_input("Enter YouTube video URL:")

        # If a URL is entered, embed the video and process it
        if youtube_url:
            # Embed the YouTube video
            st.video(youtube_url)

            # Process the YouTube video transcript as you did in your existing code
            db = create_db_from_youtube_video_url(youtube_url)

            # Ask a question about the YouTube video
            youtube_question = st.sidebar.text_input(
                "Ask a question about the YouTube video:", value=st.session_state.user_question)

            ask_youtube_button_clicked = st.sidebar.button(
                'Ask YouTube Question')

            if youtube_question and ask_youtube_button_clicked:
                st.session_state.conversation_history.append(
                    ('You', youtube_question))

                response, docs = get_response_from_query(db, youtube_question)
                st.session_state.conversation_history.append(('Bot', response))

                # Clear the text input box
                st.session_state.user_question = ""

            # Ask for a summary of the YouTube video
            summary_question = "Can you summarize this video for me? Make sure to be specific and explain the most important subjects in high detail. Use a nice layout so it's easier to understand. Please use emoji's to make it cleaner. "
            with st.expander("Summarise YouTube video"):
                summary, _ = get_response_from_query(db, summary_question)
                # st.session_state.conversation_history.append(('Bot', summary))
                st.write(summary)

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
