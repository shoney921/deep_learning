# LangChain Expression Language (LCEL)
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # OpenAI 대신 Google AI 사용
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

########################################################
# 동기 처리
########################################################
model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro",
    temperature=0.1,
)

prompt = PromptTemplate.from_template("{topic}에 대하여 3문장으로 설명해줘.")

chain = prompt | model | StrOutputParser()

# # stream : 실시간 출력
# for token in chain.stream({"topic": "쿠버네티스"}):
#     print(token, end="", flush=True)

# # invoke : 호출
# result = chain.invoke({"topic": "도커"})
# print(result)

# # batch : 배치(단위 실행)
# result = chain.batch(
#     [
#         {"topic": "mysql"},
#         {"topic": "oracle DB"},
#         {"topic": "mongodb"},
#         {"topic": "redis"}
#     ],
#     config={"max_concurrency": 2}
# )
# print(result)

########################################################
# 비동기 처리
########################################################
# async stream : 비동기 실시간 출력
async def async_stream():
    async for token in chain.astream({"topic": "스프링 부트"}):
        print(token, end="", flush=True)

# 비동기 호출 함수 생성
async def async_invoke():
    my_process = await chain.ainvoke({"topic": "NVDA"})
    print(my_process)

# async batch : 비동기 배치(단위 실행)
async def async_batch():
    result = await chain.abatch(
        [
            {"topic": "NVDA"},
            {"topic": "TSLA"},
            {"topic": "MSFT"},
            {"topic": "GOOG"},
        ],
        config={"max_concurrency": 2}
    )

# 메인 함수 정의
async def main():
    # print("\n===== 비동기 스트림 실행 =====")
    await async_stream()
    
    # print("\n\n===== 비동기 호출 실행 =====")
    await async_invoke()
    
    # print("\n\n===== 비동기 배치 실행 =====")
    await async_batch()

# 비동기 함수 실행을 위한 코드 추가
import asyncio

# 비동기 함수 실행
if __name__ == "__main__":
    asyncio.run(main())