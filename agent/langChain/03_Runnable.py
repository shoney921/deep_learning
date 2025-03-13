# LangChain Expression Language (LCEL)
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # OpenAI 대신 Google AI 사용
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

########################################################
# Runnable
########################################################
# Runnable의 장점:
# 1. 파이프라인 구성: '|' 연산자를 사용해 여러 컴포넌트를 쉽게 연결할 수 있음
# 2. 유연한 입력 처리: 다양한 형태의 입력을 자동으로 처리
# 3. 재사용성: 동일한 컴포넌트를 다양한 파이프라인에서 재사용 가능
# 4. 디버깅 용이: 각 단계별 실행 결과를 쉽게 확인 가능
# 5. 비동기 실행 지원: 대규모 병렬 처리 가능

prompt = PromptTemplate.from_template("{num}의 10배는?")
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro",
    temperature=0.1,
)

chain = prompt | llm

result = chain.invoke({"num": 10})
print(result)
# 출력 : content='10의 10배는 100입니다.' additional_kwargs={} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': []} id='run-c330367f-49bc-434a-9861-b3a821136b38-0' usage_metadata={'input_tokens': 9, 'output_tokens': 14, 'total_tokens': 23, 'input_token_details': {'cache_read': 0}}

result = chain.invoke(5)
print(result)
# 출력 : content='5의 10배는 50입니다.' additional_kwargs={} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': []} id='run-c330367f-49bc-434a-9861-b3a821136b38-1' usage_metadata={'input_tokens': 5, 'output_tokens': 14, 'total_tokens': 19, 'input_token_details': {'cache_read': 0}}


# RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough
# RunnablePassthrough의 장점:
# 1. 데이터 흐름 제어: 입력 데이터를 변형 없이 다음 단계로 전달
# 2. 데이터 구조화: 딕셔너리 형태로 데이터를 구조화하여 전달 가능
# 3. 복잡한 체인 구성: 여러 입력 소스를 조합하여 복잡한 프롬프트 구성 가능
# 4. 코드 가독성: 데이터 흐름을 명시적으로 표현하여 가독성 향상
# 5. 유지보수성: 파이프라인의 각 부분을 독립적으로 수정 가능

runnable = RunnablePassthrough()
result = runnable.invoke(6)
print(result)
# 출력 : 6

runnable_chain = {"num": runnable} | prompt | llm

result = runnable_chain.invoke({"num": 70})
print(result)
# 출력 : content='10의 10배는 100입니다.' additional_kwargs={} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': []} id='run-c330367f-49bc-434a-9861-b3a821136b38-2' usage_metadata={'input_tokens': 9, 'output_tokens': 14, 'total_tokens': 23, 'input_token_details': {'cache_read': 0}}





