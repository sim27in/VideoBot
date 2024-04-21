import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import tempfile

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from pytube import YouTube
from urllib.parse import urlparse, parse_qs
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
#youtube_api_key = os.getenv('YOUTUBE_API_KEY')

st.set_page_config(page_title="Video Genie", layout="wide")
st.markdown("""
## YouTube/Video Genie: Get Instant Insights from Your Videos
### How It Works
1. **Upload Your Video or Enter YouTube URL**: Analyzing the content to provide insights.
2. **Ask a Question**: Ask any question related to the content for a precise answer.
""")

# Define message classes
class AIMessage:
    def __init__(self, content):
        self.content = content

class HumanMessage:
    def __init__(self, content):
        self.content = content


# Helper function to extract video ID from YouTube URL
def extract_video_id(url):
    from urllib.parse import urlparse, parse_qs
    parsed_url = urlparse(url)
    if 'youtube' in parsed_url.netloc:
        return parse_qs(parsed_url.query)['v'][0]
    elif 'youtu.be' in parsed_url.netloc:
        return parsed_url.path.lstrip('/')
    return None

# Function to get vector store from URL and process YouTube transcript
def get_vectorstore_from_url(url):
    try:
        video_id = extract_video_id(url)
        if not video_id:
            return "Invalid YouTube URL"

        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        combined_transcript = " ".join([entry['text'] for entry in transcript])
        splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)
        split_text = splitter.split_text(combined_transcript)
        # Convert the document chunks to embeddings and save them to the vector store
        vectordb = FAISS.from_texts(split_text, embedding=OpenAIEmbeddings())

        return vectordb

    except TranscriptsDisabled:
        print(f"Transcript is disabled for video ID {video_id}")
        return "Transcript is disabled for this video."
    except NoTranscriptFound:
        print(f"No transcript found for video ID {video_id}")
        return "No transcript available for this video."
    except Exception as e:
        print(f"An error occurred for video ID {video_id}: {e}")
        return "An error occurred while processing the video."



# Function to get vector store from uploaded file (you'll need to define this)
def get_vectorstore_from_file(file):
    # Create a temporary file to save the uploaded file content
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        # Write the uploaded file's content to the temporary file
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Assuming `openai.api_key` is globally accessible or set as an environment variable
    system_blob = FileSystemBlobLoader(path=tmp_path)
    loader = GenericLoader(system_blob, OpenAIWhisperParser(api_key=openai.api_key))
    docs = loader.load()

    # Combine the documents' content into one string
    combined = " ".join([doc.page_content for doc in docs])

    # Split the combined text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    split_text = splitter.split_text(combined)

    # Convert the text chunks to embeddings and save them to the vector store
    vectordb = FAISS.from_texts(split_text, embedding=OpenAIEmbeddings())

    return vectordb

def get_history_aware_chain(vector_store):

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    #First we need a prompt that we can pass into an LLM to generate this search query

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
        ),
    ]
    )   
    retriever_chain = create_history_aware_retriever(llm, vector_store.as_retriever(), prompt)
    
    return retriever_chain

def get_create_stuff_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI social media expert. You specialize in Youtube and Youtube keywords. You are an expert 
            at writing descriptions, titles, and  you use keywords that help with video discovery. You are given the transcript of a video
            as context. Answer the question based on the given context:
            {context}""",
        ),
        ("system", "{chat_history}"),
        ("user", "{input}"),
    ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    return document_chain

# Function to get response (you'll need to define this)
def get_response(user_input):
    try:
        history_aware_chain = get_history_aware_chain(st.session_state.vector_store)
        document_chain = get_create_stuff_chain()
        final_chain = create_retrieval_chain(history_aware_chain, document_chain)
        
        response = final_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
        return response['answer']
    except Exception as e:
        print(f"An error occurred while generating response: {str(e)}")
        return "Sorry, I couldn't process your request."


# Sidebar for input
with st.sidebar:
    st.header("Settings")
    with st.form(key='user_input_form'):
        website_url = st.text_input("Website URL", "")
        uploaded_file = st.file_uploader("Or upload a video file", type=["mp4"])
        submit_button = st.form_submit_button(label='Submit')

# Handle form submission
if submit_button:
  with st.spinner("Processing your video...Please wait"):
    if not website_url and not uploaded_file:
        st.warning("Please enter a website URL or upload a file.")
    else:
        if website_url:
            video_id = extract_video_id(website_url)
            if video_id:
                try:
                    st.session_state.vector_store = get_vectorstore_from_url(website_url)
                    st.success("Video processed successfully.")
                except Exception as e:
                    st.error(f"Failed to process video: {str(e)}")
            else:
                st.error("Invalid YouTube URL")
        elif uploaded_file:
            # Assuming file processing is defined
            st.session_state.vector_store = get_vectorstore_from_file(uploaded_file)
            st.success("File uploaded and processed successfully.")

# Display chat interface
st.write("### Chat with AI")
user_query = st.text_input("Type your message here...", key="user_input", value="")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    if 'vector_store' in st.session_state:
        response = get_response(user_query)
        st.session_state.chat_history.append(AIMessage(content=response))
    else:
        st.session_state.chat_history.append(AIMessage(content="Please upload a video or enter a URL first."))

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
            st.info(message.content)  # AI messages with a different style
    elif isinstance(message, HumanMessage):
            st.success(message.content)  # Human messages with a different style
            
