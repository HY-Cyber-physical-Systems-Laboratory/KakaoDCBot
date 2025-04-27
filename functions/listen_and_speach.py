from typing import Optional



def listen(use_whisper: bool = False) -> str:
    """
    Listens for a command from the user and returns it as text.
    """
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print(f"Command received: {command}")
        return command
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None


def speak(text: str, use_whisper: bool = False) -> None:
    """
    Converts text to speech and plays it.
    """
    import pyttsx3

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    