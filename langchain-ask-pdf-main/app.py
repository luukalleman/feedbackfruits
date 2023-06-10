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
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.image("https://assets.website-files.com/5e318ddf83dd66053d55c38a/602ba7efd5e8b761ed988e2a_FBF%20logo%20wide.png",
             caption='Study Material Learner', use_column_width=True)

    # Added a title and explanation
    st.title("Ask Your PDF")
    st.write("""
    This app allows you to upload a PDF and ask questions about its content. 
    After you upload a PDF, it will be processed and you will be able to ask any question about the content.
    """)

    # Initialize the session state for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Moved the uploader and question input to the sidebar
    pdf = st.sidebar.file_uploader("Upload your PDF", type="pdf")

    # Use session state to store the user's question
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""
    user_question = st.sidebar.text_input(
        "Ask a question about your PDF:", value=st.session_state.user_question)

    # initialize a progress bar
    progress = st.progress(0)

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        progress.progress(10)  # Update progress

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        with st.expander("Click to see PDF file text"):
            st.write(text)

        progress.progress(20)  # Update progress

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        progress.progress(40)  # Update progress

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        progress.progress(60)  # Update progress

        # show user input and process question when 'Ask' button is clicked
        if user_question and st.sidebar.button('Ask'):
            st.session_state.conversation_history.append(
                ('You', user_question))

            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,
                                     question=user_question)
                print(cb)
            progress.progress(80)  # Update progress

            # error handling and displaying the response in a dropdown menu
            try:
                # st.sidebar.markdown(f"**Bot:** {response}")
                progress.progress(100)  # Update progress
                st.session_state.conversation_history.append(('Bot', response))
            except Exception as e:
                st.error(f"An error occurred: {e}")

            # Clear the text input box
            st.session_state.user_question = ""
            # Add a text input for YouTube URL
    youtube_url = st.sidebar.text_input("Enter YouTube video URL:")

    # If a URL is entered, embed the video and process it
    if youtube_url:
        # Embed the YouTube video
        st.video(youtube_url)

        # Process the YouTube video transcript as you did in your existing code
        db = create_db_from_youtube_video_url(youtube_url)

        # Display conversation history in the sidebar
        st.sidebar.markdown("## Conversation")
        for role, message in st.session_state.conversation_history:
            if role == 'You':
                st.sidebar.markdown(
                    f'<div style="color: #1f77b4">{role}: {message}</div>', unsafe_allow_html=True)
            else:
                st.sidebar.markdown(
                    f'<div style="color: #ff7f0e">{role}: {message}</div>', unsafe_allow_html=True)

        # Ask a question about the YouTube video
        youtube_question = st.sidebar.text_input(
            "Ask a question about the YouTube video:", value=st.session_state.user_question)

        if youtube_question and st.sidebar.button('Ask YouTube Question'):
            st.session_state.conversation_history.append(
                ('You', youtube_question))

            response, docs = get_response_from_query(db, youtube_question)
            st.session_state.conversation_history.append(('Bot', response))

            # Clear the text input box
            st.session_state.user_question = ""

    st.sidebar.markdown("## Conversation")
    for role, message in st.session_state.conversation_history:
        if role == 'Bot':
            st.sidebar.markdown(
                f'<div style="color: #6c757d;">{role}: {message}</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"{role}: {message}")


if __name__ == '__main__':
    main()
