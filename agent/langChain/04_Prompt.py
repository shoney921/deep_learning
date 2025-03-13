# LangChain Expression Language (LCEL)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Google AI 모델 초기화
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro",
    temperature=0.1,
)

# 1. 기본 PromptTemplate 예제
template = "{country}의 수도는 어디인가요?"
prompt = PromptTemplate.from_template(template)

# chain 생성 및 실행
chain = prompt | llm
response = chain.invoke({"country": "대한민국"})
print("1. 기본 프롬프트 결과:", response.content)

# 2. 다중 변수 프롬프트 예제
template_multi = "{country1}과 {country2}의 수도는 각각 어디인가요?"
prompt_multi = PromptTemplate(
    template=template_multi,
    input_variables=["country1", "country2"]
)

chain_multi = prompt_multi | llm
response_multi = chain_multi.invoke({"country1": "대한민국", "country2": "일본"})
print("\n2. 다중 변수 프롬프트 결과:", response_multi.content)

# 3. partial_variables 예제
def get_today():
    return datetime.now().strftime("%B %d")

birthday_prompt = PromptTemplate(
    template="오늘의 날짜는 {today} 입니다. 오늘이 생일인 유명인 {n}명을 나열해 주세요. 생년월일을 표기해주세요.",
    input_variables=["n"],
    partial_variables={"today": get_today}
)

chain_birthday = birthday_prompt | llm
response_birthday = chain_birthday.invoke({"n": 3})
print("\n3. 생일 프롬프트 결과:", response_birthday.content)

# 4. ChatPromptTemplate 예제
chat_template = ChatPromptTemplate.from_messages([
    # 시스템 메시지: AI의 역할과 이름을 정의합니다. {name}은 나중에 값으로 대체될 변수입니다.
    ("system", "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 {name} 입니다."),
    # 사용자(human) 메시지: 대화의 첫 번째 입력으로, 고정된 텍스트입니다.
    ("human", "반가워요!"),
    # AI 응답 메시지: AI의 첫 번째 응답으로, 고정된 텍스트입니다.
    ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
    # 사용자의 두 번째 메시지: {user_input}은 나중에 실제 사용자 입력으로 대체될 변수입니다.
    ("human", "{user_input}")
])

chain_chat = chat_template | llm
response_chat = chain_chat.invoke({
    "name": "테디",
    "user_input": "당신의 이름은 무엇입니까?"
})
print("\n4. 챗 프롬프트 결과:", response_chat.content)

# 5. MessagesPlaceholder 예제
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 요약 전문 AI 어시스턴트입니다. 당신의 임무는 주요 키워드로 대화를 요약하는 것입니다."),
    MessagesPlaceholder(variable_name="conversation"),
    ("human", "지금까지의 대화를 {word_count} 단어로 요약합니다.")
])

chain_summary = summary_prompt | llm | StrOutputParser()
response_summary = chain_summary.invoke({
    "word_count": 5,
    "conversation": [
        ("human", "안녕하세요! 저는 오늘 새로 입사한 테디 입니다. 만나서 반갑습니다."),
        ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
    ]
})
print("\n5. 대화 요약 결과:", response_summary)
