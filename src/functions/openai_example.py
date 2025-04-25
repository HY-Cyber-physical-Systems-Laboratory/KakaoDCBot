import os
import argparse
import openai

# 환경변수에서 API 키 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise RuntimeError("환경변수 OPENAI_API_KEY를 설정해주세요")

def chat_with_gpt(prompt: str) -> str:
    """
    ChatGPT API에 prompt를 전달하고 응답을 문자열로 반환합니다.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def transcribe_whisper(audio_path: str) -> str:
    """
    Whisper API에 오디오 파일을 전송하고 전사 결과를 문자열로 반환합니다.
    """
    with open(audio_path, "rb") as audio_file:
        result = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file
        )
    return result["text"].strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChatGPT 또는 Whisper를 호출하는 예제"
    )
    parser.add_argument(
        "mode",
        choices=["chat", "whisper"],
        help="chat: ChatGPT 호출, whisper: Whisper 전사"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="ChatGPT에 보낼 프롬프트 (mode=chat일 때)"
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        help="Whisper에 보낼 오디오 파일 경로 (mode=whisper일 때)"
    )
    args = parser.parse_args()

    if args.mode == "chat":
        prompt = args.prompt or input("▶ Prompt를 입력하세요: ")
        print("\n💬 ChatGPT 응답:")
        print(chat_with_gpt(prompt))

    else:  # mode == "whisper"
        audio_path = args.audio or input("▶ 오디오 파일 경로를 입력하세요: ")
        print("\n🎙 Whisper 전사 결과:")
        print(transcribe_whisper(audio_path))
