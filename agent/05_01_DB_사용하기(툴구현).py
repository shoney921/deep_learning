import os
import ssl
from langchain import hub
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine.url import URL
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

# SSL 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

# 환경 변수 설정
load_dotenv()

# 1. 기본 LLM 모델 설정
# Gemini Pro 모델을 사용하여 에이전트의 두뇌 역할을 할 LLM을 초기화
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro")

# 데이터베이스 연결 도구 클래스 추가
class DatabaseAnalysisTool:
    def __init__(self):
        # MySQL 연결 설정
        db_url = URL.create(
            drivername="mysql+pymysql",
            username="root",      
            password="root",      
            host="localhost",     
            port=3306,           
            database="sepg"      
        )
        try:
            self.engine = create_engine(db_url)
            # 연결 테스트
            with self.engine.connect() as conn:
                pass
            self.inspector = inspect(self.engine)
        except Exception as e:
            print(f"데이터베이스 연결 실패: {str(e)}")
            print("다음을 확인해주세요:")
            print("1. MySQL 서버가 실행 중인지")
            print("2. 사용자명과 비밀번호가 올바른지")
            print("3. 데이터베이스가 존재하는지")
            raise
    
    def generate_erd(self, output_path: str = "database_erd.png") -> str:
        """데이터베이스의 ERD를 생성합니다."""
        try:
            # 테이블 정보 수집
            tables = self.inspector.get_table_names()
            erd_info = []
            
            for table in tables:
                columns = self.inspector.get_columns(table)
                foreign_keys = self.inspector.get_foreign_keys(table)
                
                # 테이블 구조 정보
                table_info = f"\n테이블: {table}\n"
                table_info += "컬럼:\n"
                for col in columns:
                    table_info += f"- {col['name']}: {col['type']}\n"
                
                # 외래 키 정보
                if foreign_keys:
                    table_info += "외래 키:\n"
                    for fk in foreign_keys:
                        table_info += f"- {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}\n"
                
                erd_info.append(table_info)
            
            return "\n".join(erd_info)
        except Exception as e:
            return f"ERD 정보 수집 중 오류 발생: {str(e)}"
    
    def analyze_table(self, table_name: Optional[str] = None) -> str:
        """특정 테이블 또는 전체 데이터베이스 구조를 분석합니다."""
        try:
            if table_name:
                columns = self.inspector.get_columns(table_name)
                return f"테이블 '{table_name}'의 구조:\n" + \
                       "\n".join([f"- {col['name']}: {col['type']}" for col in columns])
            else:
                tables = self.inspector.get_table_names()
                return f"데이터베이스 테이블 목록:\n" + "\n".join([f"- {table}" for table in tables])
        except Exception as e:
            return f"테이블 분석 중 오류 발생: {str(e)}"
    
    def query_data(self, query: str) -> str:
        """SQL 쿼리를 실행하고 결과를 반환합니다."""
        try:
            df = pd.read_sql(query, self.engine)
            return df.to_string()
        except Exception as e:
            return f"쿼리 실행 중 오류 발생: {str(e)}"

# 데이터베이스 분석 도구 인스턴스 생성
db_tool = DatabaseAnalysisTool()

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
    ),
    Tool(
        name="ERD_생성",
        func=db_tool.generate_erd,
        description="데이터베이스의 ERD 다이어그램을 생성하는 도구"
    ),
    Tool(
        name="테이블_분석",
        func=db_tool.analyze_table,
        description="데이터베이스 테이블 구조를 분석하는 도구"
    ),
    Tool(
        name="SQL_쿼리",
        func=db_tool.query_data,
        description="SQL 쿼리를 실행하고 결과를 반환하는 도구"
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
    # 데이터베이스 구조 분석
    try:
        print("\n=== 데이터베이스 구조 분석 ===")
        tables = db_tool.analyze_table()
        print(tables)
        
        print("\n=== ERD 정보 ===")
        erd_info = db_tool.generate_erd()
        print(erd_info)
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    test_agent()

