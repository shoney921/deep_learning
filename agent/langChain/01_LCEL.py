# LangChain Expression Language (LCEL)
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # OpenAI 대신 Google AI 사용

load_dotenv()

########################################################
# 프롬프트 템플릿 생성
########################################################
template = "{country}의 수도는 어디인가요?"
prompt_template = PromptTemplate.from_template(template)
print(prompt_template)
# 출력 : input_variables=['country'] input_types={} partial_variables={} template='{country}의 수도는 어디인가요?'
prompt = prompt_template.format(country="한국")
print(prompt)
# 출력 : 한국의 수도는 어디인가요?
prompt = prompt_template.format(country="미국")
print(prompt)
# 출력 : 미국의 수도는 어디인가요?

########################################################
# 모델 생성
########################################################
model = ChatGoogleGenerativeAI(  # OpenAI 대신 Google AI 사용
    model="models/gemini-1.5-pro",
    temperature=0.1,
)

########################################################
# Chain 생성
########################################################
chain = prompt_template | model
print(chain.invoke({"country": "한국"}))
print(chain.invoke({"country": "미국"}))
# 출력 : content='한국의 수도는 서울입니다.' additional_kwargs={} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': []} id='run-06d6739c-0dee-4553-a13d-180aff5db85e-0' usage_metadata={'input_tokens': 11, 'output_tokens': 9, 'total_tokens': 20, 'input_token_details': {'cache_read': 0}}
# 출력 : content='미국의 수도는 워싱턴 D.C.입니다.' additional_kwargs={} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': []} id='run-06be0855-b711-4129-8386-9f51152ad87a-0' usage_metadata={'input_tokens': 11, 'output_tokens': 16, 'total_tokens': 27, 'input_token_details': {'cache_read': 0}}

########################################################
# 스트리밍 출력
########################################################
prompt_template = PromptTemplate.from_template(
    "{topic} 설명해주세요."
)
chain = prompt_template | model
input = {"topic": "인공지능 모델의 학습 원리"}
answer = chain.stream(input)
for chunk in answer:
    print(chunk.content, end="", flush=True)

########################################################
# 템플릿을 변경하여 적용
########################################################
from langchain_core.output_parsers import StrOutputParser
template = """
당신은 컴퓨터를 가르치는 교수입니다. 질문을 [FORMAT]에 맞게 설명해 주세요.

상황 : 
{question}

[FORMAT]
- 초등학생에게 알려주는 설명 :
- 대학생에게 알려주는 설명 :
"""
prompt_template = PromptTemplate.from_template(template)

model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro",
    temperature=0.1,
)

output_parser = StrOutputParser()

chain = prompt_template | model | output_parser

answer = chain.invoke({"question": "머신러닝이 뭐야?"})
print(answer)

