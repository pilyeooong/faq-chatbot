# FAQ Assistant
네이버 스마트스토어의 자주 묻는 질문(FAQ)을 기반으로 한 질의응답 챗봇 입니다.

## 사용 기술
- python 3.12
- streamlit
- chromaDB
- OpenAI
    - text-embedding-ada-002
    - gpt-4o-mini

## 실행 방법
 해당 프로젝트 실행하기 위해서는 환경 변수 설정이 먼저 필요합니다.
.env 파일을 생성해 OpenAI API를 사용할 수 있도록 `OPENAI_API_KEY`를 환경변수로 등록해주셔야 합니다.

1. pip install -r requirements.txt
2. python3 embedding.py
3. streamlit run chat.py
