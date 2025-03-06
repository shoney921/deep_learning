import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "agent/api-key.json"

from google.ai import generativelanguage_v1beta
from google.api_core import client_options

## 1. 구글에서 지원하는 모델 리스트를 먼저 출력
# 모델 리스트를 가져오기 위한 클라이언트 생성
client = generativelanguage_v1beta.ModelServiceClient()

# 사용 가능한 모델 리스트 출력
try:
    models = client.list_models()
    print("사용 가능한 모델 목록:")
    for model in models:
        print(f"- {model.name}: {model.description}")
except Exception as e:
    print("모델 목록 조회 중 오류 발생:")
    print(str(e))


## 2. 프롬프트 테스트
# 생성 모델 클라이언트 생성
client = generativelanguage_v1beta.GenerativeServiceClient()

# 테스트할 프롬프트 준비
content = generativelanguage_v1beta.Content()
content.parts = [{"text": "안녕하세요! 당신은 누구인가요?"}]

# 요청 생성
request = generativelanguage_v1beta.GenerateContentRequest(
    model="models/gemini-1.5-pro",
    contents=[content]
)

try:
    response = client.generate_content(request)
    print("\n응답:")
    print(response.candidates[0].content.parts[0].text)
except Exception as e:
    print("오류 발생:")
    print(str(e))

