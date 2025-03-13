# LangChain Expression Language (LCEL)
from langchain_core.prompts import (
    ChatPromptTemplate, 
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import  JsonOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# Google AI 모델 초기화
# temperature: 0에 가까울수록 일관된 응답, 1에 가까울수록 창의적인 응답
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro",
    temperature=0.1,
)

# Pydantic을 사용하여 원하는 출력 형식을 정의
# BaseModel을 상속받아 클래스를 정의하면 데이터 유효성 검사와 직렬화가 자동으로 처리됨
class Topic(BaseModel):
    # Field는 각 필드에 대한 메타데이터를 정의할 때 사용
    # description 파라미터로 AI 모델에게 각 필드가 어떤 내용을 담아야 하는지 설명
    description: str = Field(description="주제에 대한 간결한 설명")
    hashtags: str = Field(description="해시태그 형식의 키워드(2개 이상)")

# JsonOutputParser 초기화 - Topic 클래스의 형식에 맞춰 JSON 출력을 생성
parser = JsonOutputParser(pydantic_object=Topic)

# 프롬프트 템플릿 생성
# {format_instructions}: parser가 제공하는 출력 형식 지침
# {question}: 사용자의 질문이 들어갈 자리
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트입니다. 질문에 간결하게 답변하세요."),
    ("user", "#Format: {format_instructions}\n\n#Question: {question}"),
])

# format_instructions를 미리 설정 (partial 적용)
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# LCEL을 사용하여 체인 구성: prompt -> llm -> parser
chain = prompt | llm | parser

# 체인 실행 예시
if __name__ == "__main__":
    # 테스트 질문
    question = "한국의 2025년 경제 현황에 대해 알려주세요."
    
    # 체인 실행
    response = chain.invoke({"question": question})
    print("\n=== AI 응답 ===")
    print(response)
    # print(f"설명: {response['description']}")
    # print(f"해시태그: {response['hashtags']}")

