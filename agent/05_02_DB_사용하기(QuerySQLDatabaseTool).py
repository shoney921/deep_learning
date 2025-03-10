import os
import ssl
from langchain import hub
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv

# SSL 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수는 이제 .env 파일에서 자동으로 로드됩니다
# os.environ 직접 설정 부분 제거

# 1. 기본 LLM 모델 설정
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro")

# 2. MySQL 데이터베이스 연결
db = SQLDatabase.from_uri(
    "mysql+pymysql://root:root@localhost:3306/sepg"
)

# 3. 도구(Tools) 정의
tools = [
    Tool(
        name="웹검색",
        func=SerpAPIWrapper().run,
        description="최신 정보를 검색해야 할 때 사용하는 도구"
    ),
    PythonREPLTool(
        name="파이썬_실행기",
        description="파이썬 코드를 실행해야 할 때 사용하는 도구"
    ),
    QuerySQLDatabaseTool(
        db=db,
        name="데이터베이스_조회",
        description="MySQL 데이터베이스에 대한 질의를 실행할 때 사용하는 도구. 테이블 구조 확인, 데이터 조회 등이 가능합니다."
    )
]

# 4. 프롬프트 템플릿 가져오기
prompt = hub.pull("hwchase17/react")

# 5. ReAct 에이전트 생성
agent = create_react_agent(llm, tools, prompt)

# 6. 에이전트 실행기 생성
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True
)

# 7. 에이전트 테스트
def test_agent():
    # 단일 테이블 구조 조회로 시작
    result = agent_executor.invoke(
        {"input": "mysql의 데이터베이스중에 sepg데이터베이스안의 테이블들을 보고 어떤 데이터베이스인지 요약해줘"}
    )
    print("\n테이블 구조:", result["output"])

def test_advanced_queries():
    # 데이터 분석 쿼리
    analysis_query = "sepg 데이터베이스의 데이터를 분석해서 주요 통계 정보를 추출해줘. 테이블간의 관계도 파악해서 설명해줘"
    
    # 데이터 시각화 요청
    visualization_query = "sepg 데이터베이스의 주요 데이터를 파이썬으로 시각화해서 보여줘"
    
    # 비즈니스 인사이트 도출
    insight_query = "데이터를 기반으로 어떤 비즈니스 인사이트를 도출할 수 있을지 분석해줘"
    
    results = []
    for query in [analysis_query, visualization_query, insight_query]:
        result = agent_executor.invoke({"input": query})
        results.append(result["output"])
        print(f"\n실행 결과: {result['output']}\n")

def test_conversational_queries():
    # 첫 번째 쿼리
    initial_query = "sepg 데이터베이스에서 가장 중요한 테이블이 무엇인지 찾아줘"
    result1 = agent_executor.invoke({"input": initial_query})
    
    # 이전 결과를 기반으로 한 후속 쿼리
    followup_query = f"방금 찾은 테이블의 데이터를 자세히 분석해서 특이사항을 알려줘"
    result2 = agent_executor.invoke({"input": followup_query})

if __name__ == "__main__":
    # test_agent()
    # test_advanced_queries()
    test_conversational_queries()

