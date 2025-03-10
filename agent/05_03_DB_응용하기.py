"""
Langchain을 활용한 데이터베이스 분석 에이전트
이 스크립트는 Gemini Pro LLM과 MySQL 데이터베이스를 연동하여
데이터 분석과 인사이트 도출을 자동화하는 에이전트를 구현합니다.

작성자: [작성자명]
작성일: [날짜]
"""

import os
import ssl
from typing import List
from dotenv import load_dotenv
from langchain import hub
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain.tools import tool

class DatabaseAnalysisAgent:
    """데이터베이스 분석을 위한 AI 에이전트 클래스"""
    
    def __init__(self):
        """에이전트 초기화 및 필요한 컴포넌트 설정"""
        # SSL 인증 설정
        ssl._create_default_https_context = ssl._create_unverified_context
        load_dotenv()
        
        # LLM 모델 초기화
        self.llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro")
        
        # 데이터베이스 연결
        self.db = SQLDatabase.from_uri(
            "mysql+pymysql://root:root@localhost:3306/sepg"
        )
        
        # SQL 쿼리 생성 체인 추가
        self.sql_chain = create_sql_query_chain(
            self.llm,
            self.db,
            k=5  # top-k 유사 예제 수
        )
        
        # 에이전트 설정
        self.tools = self._setup_tools()
        self.agent_executor = self._create_agent()

    def _setup_tools(self) -> List[Tool]:
        """에이전트가 사용할 도구 설정"""
        basic_tools = [
            Tool(
                name="웹검색",
                func=SerpAPIWrapper().run,
                description="실시간 웹 정보 검색용 도구"
            ),
            PythonREPLTool(
                name="파이썬_실행기",
                description="파이썬 코드 실행 및 데이터 처리용 도구"
            ),
            QuerySQLDatabaseTool(
                db=self.db,
                name="데이터베이스_조회",
                description="MySQL 데이터베이스 쿼리 실행 및 스키마 분석용 도구"
            )
        ]
        
        # 커스텀 도구 추가
        custom_tools = [
            Tool(
                name="자연어_SQL_변환",
                func=self.natural_to_sql,
                description="자연어를 SQL 쿼리로 변환하는 도구"
            ),
            Tool(
                name="데이터_시각화",
                func=self.visualize_data,
                description="SQL 쿼리 결과를 시각화하는 도구"
            )
        ]
        
        return basic_tools + custom_tools

    def _create_agent(self) -> AgentExecutor:
        """ReAct 에이전트 생성"""
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    @tool
    def natural_to_sql(self, query: str) -> str:
        """자연어를 SQL 쿼리로 변환"""
        sql_query = self.sql_chain.invoke({"question": query})
        return f"생성된 SQL 쿼리: {sql_query}"

    @tool
    def visualize_data(self, sql_query: str, chart_type: str = "bar") -> str:
        """SQL 쿼리 결과를 시각화"""
        try:
            # SQL 실행 및 DataFrame 변환
            result = self.db.run(sql_query)
            df = pd.DataFrame([dict(row) for row in result])
            
            # 시각화 유형에 따른 처리
            if chart_type == "bar":
                fig = px.bar(df)
            elif chart_type == "line":
                fig = px.line(df)
            elif chart_type == "scatter":
                fig = px.scatter(df)
            elif chart_type == "pie":
                fig = px.pie(df)
            else:
                return "지원하지 않는 차트 유형입니다."
            
            # HTML 파일로 저장
            output_path = "visualization.html"
            fig.write_html(output_path)
            return f"시각화가 {output_path}에 저장되었습니다."
            
        except Exception as e:
            return f"시각화 중 오류 발생: {str(e)}"

    def analyze_database_structure(self):
        """데이터베이스 구조 분석"""
        return self.agent_executor.invoke({
            "input": "mysql의 데이터베이스중에 sepg데이터베이스안의 테이블들을 보고 어떤 데이터베이스인지 요약해줘"
        })

    def perform_advanced_analysis(self):
        """심화 데이터 분석 수행"""
        queries = [
            "sepg 데이터베이스의 데이터를 분석해서 주요 통계 정보를 추출해줘. 테이블간의 관계도 파악해서 설명해줘",
            "sepg 데이터베이스의 주요 데이터를 시각화해서 보여줘. 막대 그래프와 파이 차트로 표현해줘",
            "데이터를 기반으로 어떤 비즈니스 인사이트를 도출할 수 있을지 분석해줘",
            "각 테이블의 데이터 분포를 시각화해서 보여줘"
        ]
        
        results = []
        for query in queries:
            result = self.agent_executor.invoke({"input": query})
            results.append(result["output"])
        return results

    def interactive_analysis(self):
        """대화형 데이터 분석 수행"""
        initial_result = self.agent_executor.invoke({
            "input": "sepg 데이터베이스에서 가장 중요한 테이블이 무엇인지 찾아줘"
        })
        
        followup_result = self.agent_executor.invoke({
            "input": "방금 찾은 테이블의 데이터를 자세히 분석해서 특이사항을 알려줘"
        })
        
        return initial_result, followup_result

def main():
    """메인 실행 함수"""
    agent = DatabaseAnalysisAgent()
    
    interactive_results = agent.interactive_analysis()

if __name__ == "__main__":
    main()

