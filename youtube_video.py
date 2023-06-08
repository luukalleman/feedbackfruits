from flask import Flask, request, jsonify
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
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

app = Flask(__name__)

load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()

def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    video_url = data.get('video_url')
    question = data.get('question')
    if not video_url or not question:
        return jsonify({'error': 'Missing video_url or question'}), 400

    db = create_db_from_youtube_video_url(video_url)
    response, docs = get_response_from_query(db, question)
    return jsonify({'response': response, 'docs': docs}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)

# from langchain.document_loaders import YoutubeLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from dotenv import find_dotenv, load_dotenv
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )
# import textwrap
# import os

# load_dotenv(find_dotenv())
# # Now the environment variables are loaded, you can access them like this:
# openai_api_key = os.getenv("OPENAI_API_KEY")

# # Now you can use this key in your application. For example:
# embeddings = OpenAIEmbeddings()


# def create_db_from_youtube_video_url(video_url):
#     loader = YoutubeLoader.from_youtube_url(video_url)
#     transcript = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     docs = text_splitter.split_documents(transcript)

#     db = FAISS.from_documents(docs, embeddings)
#     return db


# def get_response_from_query(db, query, k=4):
#     """
#     gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
#     the number of tokens to analyze.
#     """

#     docs = db.similarity_search(query, k=k)
#     docs_page_content = " ".join([d.page_content for d in docs])

#     chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

#     # Template to use for the system message prompt
#     template = """
#         You are a helpful assistant that that can answer questions about youtube videos 
#         based on the video's transcript: {docs}
        
#         Only use the factual information from the transcript to answer the question.
        
#         If you feel like you don't have enough information to answer the question, say "I don't know".
        
#         Your answers should be verbose and detailed.
#         """

#     system_message_prompt = SystemMessagePromptTemplate.from_template(template)

#     # Human question prompt
#     human_template = "Answer the following question: {question}"
#     human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [system_message_prompt, human_message_prompt]
#     )

#     chain = LLMChain(llm=chat, prompt=chat_prompt)

#     response = chain.run(question=query, docs=docs_page_content)
#     response = response.replace("\n", "")
#     return response, docs


# # Example usage:
# video_url = "https://www.youtube.com/watch?v=dN0lsF2cvm4"
# db = create_db_from_youtube_video_url(video_url)

# query = "What is this about?"
# response, docs = get_response_from_query(db, query)
# print(textwrap.fill(response, width=50))