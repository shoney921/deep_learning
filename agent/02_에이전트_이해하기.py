import os
import ssl
from langchain import hub
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor

# SSL 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

# 환경 변수 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "agent/api-key.json"
os.environ["SERPAPI_API_KEY"] = "7326a483d8e7f2a32492627bd1f4de6df90b52deb1ea77be91ec17be6cd1df3b"  # https://serpapi.com 에서 무료 API 키 발급
os.environ['PYTHONHTTPSVERIFY'] = '0'

# 1. 기본 LLM 모델 설정
# Gemini Pro 모델을 사용하여 에이전트의 두뇌 역할을 할 LLM을 초기화
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro")

# 2. 도구(Tools) 정의
# 에이전트가 사용할 수 있는 도구들을 정의
tools = [
    Tool(
        name="웹검색",
        func=SerpAPIWrapper().run,
        description="최신 정보를 검색해야 할 때 사용하는 도구"
    ),
    PythonREPLTool(
        name="파이썬_실행기",
        description="파이썬 코드를 실행해야 할 때 사용하는 도구"
    )
]

# 3. 프롬프트 템플릿 가져오기
# ReAct 프레임워크를 사용하는 에이전트를 위한 프롬프트 템플릿
prompt = hub.pull("hwchase17/react")

# 4. ReAct 에이전트 생성
# LLM, 도구, 프롬프트를 조합하여 에이전트 생성
agent = create_react_agent(llm, tools, prompt)

# 5. 에이전트 실행기 생성
# 실제로 에이전트를 실행할 수 있는 실행기 생성
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. 에이전트 테스트
def test_agent():
    # 파이썬 코드 실행 테스트 - 간단한 계산
    result1 = agent_executor.invoke(
        {"input": "파이썬을 사용해서 1부터 10까지의 숫자 중 짝수만 출력해주세요"}
    )
    print("\n기본 계산 결과:", result1["output"])
    
    # 파이썬 코드 실행 테스트 - 데이터 분석
    result2 = agent_executor.invoke(
        {"input": """다음 데이터의 평균과 표준편차를 계산해주세요: 
        [23, 45, 67, 89, 12, 34, 56, 78, 90, 11]"""}
    )
    print("\n통계 분석 결과:", result2["output"])
    
    # 파이썬 코드 실행 테스트 - 문자열 처리
    result3 = agent_executor.invoke(
        {"input": "문자열 'Hello, Python World!'에서 모든 모음(a,e,i,o,u)의 개수를 세어주세요"}
    )
    print("\n문자열 분석 결과:", result3["output"])
    
    # 복합 작업 테스트 - 웹 검색 결과를 파이썬으로 처리
    result4 = agent_executor.invoke(
        {"input": "대한민국의 2023년 GDP가 얼마인지 검색하고, 이를 달러에서 원화로 환산해주세요 (1달러=1300원 기준)"}
    )
    print("\n복합 작업 결과:", result4["output"])

if __name__ == "__main__":
    test_agent()

