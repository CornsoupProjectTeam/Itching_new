import os
from dotenv import load_dotenv
from chatbot import start_chatbot
from context_rag_cloud import interactive_process, batch_process

def main():
    print("=== BIG 5 성격 분석 프로그램 ===")
    mode = input("모드를 선택하세요 ('interactive' 또는 'chatting'): ").strip().lower()

    if mode == "interactive":
        print("Interactive 모드 실행 중...")
        interactive_process()
    elif mode == "chatting":
        print("Chatting 모드 실행 중...")
        print("Chatbot 실행 중...")
        start_chatbot()  # chatbot.py 실행 후 chat_archive.json 생성
        print("Chatbot 실행 완료. 분석 시작...")
        batch_process("chat_archives.json", "results.json")
    else:
        print("올바른 모드를 선택해주세요 ('interactive' 또는 'batch').")


if __name__ == "__main__":
    main()
