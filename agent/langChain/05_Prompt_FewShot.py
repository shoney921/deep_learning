# LangChain Expression Language (LCEL)
from langchain_core.prompts import (
    ChatPromptTemplate, 
    FewShotChatMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Google AI 모델 초기화
# temperature: 0에 가까울수록 일관된 응답, 1에 가까울수록 창의적인 응답
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro",
    temperature=0.1,
)

# Few-shot 학습을 위한 예제 데이터
# instruction: AI에게 주는 지시사항
# input: 실제 입력 데이터
# answer: 기대하는 응답 형식
examples = [
    {
        "instruction": "당신은 회의록 작성 전문가 입니다. 주어진 정보를 바탕으로 회의록을 작성해 주세요",
        "input": "2023년 12월 25일, XYZ 회사의 마케팅 전략 회의가 오후 3시에 시작되었다. 회의에는 마케팅 팀장인 김수진, 디지털 마케팅 담당자인 박지민, 소셜 미디어 관리자인 이준호가 참석했다. 회의의 주요 목적은 2024년 상반기 마케팅 전략을 수립하고, 새로운 소셜 미디어 캠페인에 대한 아이디어를 논의하는 것이었다. 팀장인 김수진은 최근 시장 동향에 대한 간략한 개요를 제공했으며, 이어서 각 팀원이 자신의 분야에서의 전략적 아이디어를 발표했다.",
        "answer": """
회의록: XYZ 회사 마케팅 전략 회의
일시: 2023년 12월 25일
장소: XYZ 회사 회의실
참석자: 김수진 (마케팅 팀장), 박지민 (디지털 마케팅 담당자), 이준호 (소셜 미디어 관리자)

1. 개회
   - 회의는 김수진 팀장의 개회사로 시작됨.
   - 회의의 목적은 2024년 상반기 마케팅 전략 수립 및 새로운 소셜 미디어 캠페인 아이디어 논의.

2. 시장 동향 개요 (김수진)
   - 김수진 팀장은 최근 시장 동향에 대한 분석을 제시.
   - 소비자 행동 변화와 경쟁사 전략에 대한 통찰 공유.

3. 디지털 마케팅 전략 (박지민)
   - 박지민은 디지털 마케팅 전략에 대해 발표.
   - 온라인 광고와 SEO 최적화 방안에 중점을 둠.

4. 소셜 미디어 캠페인 (이준호)
   - 이준호는 새로운 소셜 미디어 캠페인에 대한 아이디어를 제안.
   - 인플루언서 마케팅과 콘텐츠 전략에 대한 계획을 설명함.

5. 종합 논의
   - 팀원들 간의 아이디어 공유 및 토론.
   - 각 전략에 대한 예산 및 자원 배분에 대해 논의.

6. 마무리
   - 다음 회의 날짜 및 시간 확정.
   - 회의록 정리 및 배포는 박지민 담당.
""",
    },
    {
        "instruction": "당신은 요약 전문가 입니다. 다음 주어진 정보를 바탕으로 내용을 요약해 주세요",
        "input": "이 문서는 '지속 가능한 도시 개발을 위한 전략'에 대한 20페이지 분량의 보고서입니다. 보고서는 지속 가능한 도시 개발의 중요성, 현재 도시화의 문제점, 그리고 도시 개발을 지속 가능하게 만들기 위한 다양한 전략을 포괄적으로 다루고 있습니다. 이 보고서는 또한 성공적인 지속 가능한 도시 개발 사례를 여러 국가에서 소개하고, 이러한 사례들을 통해 얻은 교훈을 요약하고 있습니다.",
        "answer": """
문서 요약: 지속 가능한 도시 개발을 위한 전략 보고서

- 중요성: 지속 가능한 도시 개발이 필수적인 이유와 그에 따른 사회적, 경제적, 환경적 이익을 강조.
- 현 문제점: 현재의 도시화 과정에서 발생하는 주요 문제점들, 예를 들어 환경 오염, 자원 고갈, 불평등 증가 등을 분석.
- 전략: 지속 가능한 도시 개발을 달성하기 위한 다양한 전략 제시. 이에는 친환경 건축, 대중교통 개선, 에너지 효율성 증대, 지역사회 참여 강화 등이 포함됨.
- 사례 연구: 전 세계 여러 도시의 성공적인 지속 가능한 개발 사례를 소개. 예를 들어, 덴마크의 코펜하겐, 일본의 요코하마 등의 사례를 통해 실현 가능한 전략들을 설명.
- 교훈: 이러한 사례들에서 얻은 주요 교훈을 요약. 강조된 교훈에는 다각적 접근의 중요성, 지역사회와의 협력, 장기적 계획의 필요성 등이 포함됨.

이 보고서는 지속 가능한 도시 개발이 어떻게 현실적이고 효과적인 형태로 이루어질 수 있는지에 대한 심도 있는 분석을 제공합니다.
""",
    },
    {
        "instruction": "당신은 문장 교정 전문가 입니다. 다음 주어진 문장을 교정해 주세요",
        "input": "우리 회사는 새로운 마케팅 전략을 도입하려고 한다. 이를 통해 고객과의 소통이 더 효과적이 될 것이다.",
        "answer": "본 회사는 새로운 마케팅 전략을 도입함으로써, 고객과의 소통을 보다 효과적으로 개선할 수 있을 것으로 기대된다.",
    },
]

# 예제 프롬프트 템플릿 생성
# ChatPromptTemplate.from_messages()를 사용하여 대화형 프롬프트 생성
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{instruction}:\n{input}"),  # 사용자 입력 형식
    ("ai", "{answer}")  # AI 응답 형식
])

# SemanticSimilarityExampleSelector 설정
# 입력된 질문과 가장 유사한 예제를 선택하기 위한 선택기
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    GoogleGenerativeAIEmbeddings(model="models/embedding-001"),  # Google 임베딩 모델 사용
    Chroma,
    k=1
)

# FewShotChatMessagePromptTemplate 생성
# 선택된 예제를 기반으로 few-shot 학습 프롬프트 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
)

# 최종 프롬프트 템플릿 생성
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 전문적인 AI 어시스턴트입니다."),  # 시스템 메시지로 AI의 역할 정의
    few_shot_prompt,  # few-shot 예제들
    ("human", "{instruction}\n{input}")  # 실제 사용자 입력
])

# 테스트를 위한 새로운 입력
new_input = {
    "instruction": "회의록을 작성해 주세요",
    "input": "2023년 12월 26일, ABC 기술 회사의 제품 개발 팀은 새로운 모바일 애플리케이션 프로젝트에 대한 주간 진행 상황 회의를 가졌다. 이 회의에는 프로젝트 매니저인 최현수, 주요 개발자인 황지연, UI/UX 디자이너인 김태영이 참석했다. 회의의 주요 목적은 프로젝트의 현재 진행 상황을 검토하고, 다가오는 마일스톤에 대한 계획을 수립하는 것이었다. 각 팀원은 자신의 작업 영역에 대한 업데이트를 제공했고, 팀은 다음 주까지의 목표를 설정했다.",
}

# 체인 생성 및 실행
chain = final_prompt | llm | StrOutputParser()
response = chain.invoke(new_input)
print("생성된 회의록:\n", response)

# # 다른 유형의 입력 테스트
# summary_input = {
#     "instruction": "문서를 요약해 주세요",
#     "input": """2024년 1분기 시장 분석 보고서에 따르면, 
#     모바일 결제 시장이 전년 대비 25% 성장했으며, 
#     특히 MZ세대의 디지털 월렛 사용이 크게 증가한 것으로 나타났다. 
#     이는 비대면 거래 선호도 증가와 간편 결제 서비스의 확대가 주요 원인으로 분석된다."""
# }

# response = chain.invoke(summary_input)
# print("\n문서 요약 결과:\n", response)

