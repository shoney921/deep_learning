import os
import ssl
import warnings
from langchain import hub
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, create_structured_chat_agent, create_openai_functions_agent, AgentExecutor
from langchain_community.tools import (
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
    ShellTool,
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv

# SSL 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

# 환경 변수 설정
load_dotenv()

# LangSmith 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith.client")

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
    Tool(
        name="덕덕고_검색",
        func=DuckDuckGoSearchRun().run,
        description="SerpAPI의 대안으로 무료로 웹 검색을 할 수 있는 도구"
    ),
    Tool(
        name="위키피디아",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
        description="위키피디아에서 정보를 검색해야 할 때 사용하는 도구"
    ),
    PythonREPLTool(
        name="파이썬_실행기",
        description="파이썬 코드를 실행해야 할 때 사용하는 도구"
    ),
    ShellTool(
        name="터미널_명령어",
        description="시스템 명령어를 실행해야 할 때 사용하는 도구"
    ),
    ReadFileTool(
        name="파일_읽기",
        description="파일의 내용을 읽어야 할 때 사용하는 도구"
    ),
    WriteFileTool(
        name="파일_쓰기",
        description="파일에 내용을 써야 할 때 사용하는 도구"
    ),
    ListDirectoryTool(
        name="디렉토리_목록",
        description="디렉토리의 파일 목록을 확인해야 할 때 사용하는 도구"
    ),
]

# 3. 프롬프트 템플릿 가져오기
prompt = hub.pull("hwchase17/structured-chat-agent")
agent = create_structured_chat_agent(llm, tools, prompt)


# 4. 에이전트 실행기 생성
# 실제로 에이전트를 실행할 수 있는 실행기 생성
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True  # 파싱 에러 처리 옵션 추가
)

# 5. 에이전트 테스트
def test_agent():
    try:
        # 위키피디아 검색 테스트
        result1 = agent_executor.invoke(
            {"input": "1부터 100까지의 숫자 중 소수만 계산해서 'prime_numbers.txt' 파일에 저장해주세요"}
        )
        print("\n위키피디아 검색 결과:", result1["output"])
    except Exception as e:
        print("위키피디아 검색 중 에러 발생:", str(e))
    
    try:
        # 파일 목록 테스트
        result3 = agent_executor.invoke(
            {"input": "현재 디렉토리의 파일을 보고 어떤 디렉토리의 주제가 뭔지 알려줘"}
        )
        print("\n디렉토리 목록:", result3["output"])
    except Exception as e:
        print("디렉토리 목록 조회 중 에러 발생:", str(e))

if __name__ == "__main__":
    test_agent()

