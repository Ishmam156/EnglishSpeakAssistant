import config
from google.cloud import texttospeech
import gradio as gr
import openai
import os
from playsound import playsound

openai.api_key = config.OPENAI_API_KEY
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./cloud_creds.json"

text_for_speech = ""

ROLE_MAPPING = {"assistant": "Your English Assistant", "user": "You"}
messages = [
    {
        "role": "system",
        "content": """
            You are a highly skilled AI trained English Speaking Coach. 
            Users will be speaking to you in simple English and I want you to hold conversations with them. 
            Make sure you always use simple language and don't make your responses very lengthy. 
            Always try to steer the conversation to keep it going and try to get the speaker to give you clues about their day and life so that you can ask further questions. 
            Speakers will be of varying age and their English proficiency will be very basic. 
            Never tell the user that you are an AI as the user should feel like they are conversing with a human and if needed, 
            deflect answers to avoid mentioning you are an AI and give answers that an average human might likely give at that time.
            """,
    }
]


def convert_messages_to_text():
    global messages

    chat_transcript = ""
    for message in messages:
        if message["role"] != "system":
            chat_transcript += (
                ROLE_MAPPING[message["role"]] +
                ": " + message["content"] + "\n\n"
            )

    return chat_transcript


def play_audio():
    global text_for_speech

    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text_for_speech)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

    playsound("./output.mp3")
    convert_messages_to_text()

    return None


def get_response_from_chatgpt(text):
    global text_for_speech, messages

    messages.append({"role": "user", "content": text})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages)

    system_message = response["choices"][0]["message"]

    messages.append(system_message)

    chat_transcript = convert_messages_to_text()

    text_for_speech = system_message["content"]

    return chat_transcript


def get_text_from_speech(audio_path):
    if not audio_path:
        return None

    with open(audio_path, "rb") as audio_file:
        transcription = openai.Audio.transcribe("whisper-1", audio_file)

    return transcription["text"]


def transcribe(audio):
    transcript = get_text_from_speech(audio)
    if not transcript:
        return "Kindly re-record your voice again!"

    chat_transcript = get_response_from_chatgpt(transcript)

    play_audio()

    return chat_transcript


def clear():
    return None


def clear_message():
    global text_for_speech, messages

    text_for_speech = ""

    messages = [
        {
            "role": "system",
            "content": """
                You are a highly skilled AI trained English Speaking Coach. 
                Users will be speaking to you in simple English and I want you to hold conversations with them. 
                Make sure you always use simple language and don't make your responses very lengthy. 
                Always try to steer the conversation to keep it going and try to get the speaker to give you clues about their day and life so that you can ask further questions. 
                Speakers will be of varying age and their English proficiency will be very basic. 
                Never tell the user that you are an AI as the user should feel like they are conversing with a human and if needed, 
                deflect answers to avoid mentioning you are an AI and give answers that an average human might likely give at that time.
                """,
        }
    ]

    return None


with gr.Blocks(
    title="English Speaking Assistant!", css="h1, p {text-align: center}"
) as demo:
    gr.Markdown(
        """
    # Your English Speaking Assistant!
    Practice speaking English by having a fluent conversation with AI!
    """
    )
    audio = gr.Audio(source="microphone", type="filepath",
                     label="Record your voice.")

    submit_button = gr.Button("Submit your audio recording!")

    output = gr.Textbox(
        label="Response from AI",
        placeholder="Response will appear here..",
    )

    submit_button.click(fn=transcribe, inputs=audio, outputs=output)

    clear_btn = gr.Button(value="Clear previous audio!")
    clear_btn.click(clear, inputs=[], outputs=[audio])
    clear_text_btn = gr.Button(value="Clear previous messages!")
    clear_text_btn.click(clear_message, inputs=[], outputs=[audio])

# demo.launch(share=True) # use for shareable link
demo.launch()
