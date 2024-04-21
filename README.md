# Video Genie

Video Genie is a Streamlit application that provides instant insights from YouTube videos or uploaded video files. It leverages OpenAI's Whisper model for transcription and further allows users to interact with the transcribed content through a chat interface powered by OpenAI's language model.

## Features

- Upload MP4 video files or provide a YouTube URL for processing.
- Get transcripts of the video content.
- Chat with an AI interface to ask questions and gain insights based on the video content.

## How to Install

Make sure you have Python 3.8 or newer installed on your system. To install Video Genie, follow these steps:

1. Clone the repository to your local machine:

git clone https://github.com/sim27in/VideoBot.git
cd your-repository-directory

2. Install the required dependencies:

   pip install -r requirements.txt

3. Set up your OpenAI API key as an environment variable:

export OPENAI_API_KEY='your_openai_api_key_here'

4. How to Use
   
To run the Video Genie Streamlit app, execute the following command in the terminal:
   
   streamlit run yt_transcribe.py

## Input
You can input a YouTube URL directly into the provided text field.
Alternatively, you can upload a video file from your computer.

##Interaction
Once the video is processed, start interacting with the AI by typing your questions into the chat interface.
The AI will provide responses based on the content of the video.

##License
This project is licensed under the MIT License - see the LICENSE.md file for details.

##Acknowledgments
Thanks to the OpenAI team for providing the powerful Whisper model and GPT-3 API.
Streamlit for enabling rapid development of data applications.

