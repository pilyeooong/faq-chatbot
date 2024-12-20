from dotenv import load_dotenv
from embedding import get_retriever
import os
from openai import OpenAI

load_dotenv()

def get_llm_client():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    return client

def get_ai_response(message_list, question):
    client = get_llm_client()
    retrieved = get_retriever(question)
    context = "\n".join(retrieved)
    prompt = f"""당신은 스마트스토어 FAQ 담당 AI입니다.
    아래 문서를 참고하여 질문에 정확하게 답변해주세요. 

    1. **사용자의 입력이 명확한 요청형 문장일 경우**: 해당 질문에 대해 문서 기반으로 직접 답변하세요.
    2. **사용자의 입력이 단순 키워드 또는 불명확한 질문일 경우**: 사용자의 의도를 추론하여 적절하고 구체적인 답변을 제공해주세요.
    - 예를 들어, 'API 사용법' 같은 입력은 'API 사용법에 대해 알려주세요'라는 요청으로 해석하여 답변해주세요.

    3. 스마트 스토어와 관련 없는 질문에는 다음과 같이 답변하세요:  
    "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."

    질문에 대한 답변 이후, 사용자의 질문과 키워드를 기반으로 이어질 수 있는 관련 질문을 최대 3개 추천해주세요.

    ### 출력 형식:
    1. **답변**: 질문에 대한 문서 기반 답변
    2. **추천 질문**: 질문과 이어질 수 있는 관련 질문 3개를 리스트로 제공

    ### 추천 질문 조건:
    1. 현재 질문과 가장 유사한 주제의 질문  
    2. 문서에서 다루고 있지만 더 깊게 탐색할 수 있는 질문  
    3. 사용자에게 실용적으로 도움이 될 추가적인 질문  

    ### 출력 예시:
    1. **답변**:
    스마트 스토어에서 API를 활용하는 방법은 다음과 같습니다. API 키를 발급받고 문서에 따라 요청 및 응답 형식을 설정하면 됩니다.

    2. **추천 질문**:
    - 스마트 스토어 API 키는 어디서 발급받나요?
    - API 호출 예제 코드가 있나요?
    - API 에러가 발생하면 어떻게 해결하나요?

    ### 문서:
    {context}

    ### 질문:
    {question}

    ### 출력:"""


    messages = message_list + [
        {"role": "system", "content": "당신은 스마트스토어 FAQ 담당 AI입니다."},
        {"role": "user", "content": prompt }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    def stream_response():
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content  

    return stream_response()
