# 스타트업 투자 회의록 Q&A Agent

이 프로젝트는 스타트업 투자 회의록 데이터를 기반으로 Q&A를 지원하는 AI Agent입니다. LangChain, LangGraph, Langfuse 등 최신 LLM 오케스트레이션 기술을 활용하며, RAG(Retrieval-Augmented Generation)와 websearch tool을 지원합니다.

## 주요 특징
- **LangChain, LangGraph, Langfuse** 기반의 LLM Q&A 에이전트
- **RAG**(회의록 벡터 검색) 및 **websearch**(실시간 웹 검색) tool 지원
- 스타트업 투자 회의록 JSON 데이터 기반 질의응답
- Streamlit 기반 웹 UI

## 기술 스택
- Python 3.10 이상
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Langfuse](https://github.com/langfuse/langfuse)
- Streamlit

## 설치 방법

1. 저장소 클론
   ```bash
   git clone <YOUR_REPO_URL>
   cd invest-bot
   ```

2. 패키지 설치 (uv 사용)
   ```bash
   uv pip install -r pyproject.toml
   ```
   또는 poetry/pip 등 Python 표준 패키지 매니저로도 설치 가능합니다.

3. 환경 변수 설정
   - `.env` 파일을 생성하여 OpenAI, Langfuse 등 필요한 API 키를 입력하세요.
   - 예시:
     ```env
     OPENAI_API_KEY=sk-...
     LANGFUSE_PUBLIC_KEY=...
     LANGFUSE_SECRET_KEY=...
     LANGFUSE_HOST=https://cloud.langfuse.com
     ```

4. 회의록 데이터 준비
   - `data/startup_investment_meeting_data_30.json` 파일 예시:
     ```json
     [
       {
         "id": "meeting_1",
         "title": "시리즈 A 준비 미팅",
         "content": "질문: ...\n답변: ...\n참석자: ...\n회의 일시: ...\n"
       },
       ...
     ]
     ```

## 실행 방법

```bash
streamlit run main.py
```

실행 후 웹 브라우저에서 Q&A 인터페이스를 사용할 수 있습니다.

## 폴더 구조

```
.
├── main.py                # Streamlit 앱 진입점
├── pyproject.toml         # 의존성 및 프로젝트 설정
├── data/                  # 회의록 JSON 데이터
├── chroma_db/             # 벡터 DB 저장소 (자동 생성)
└── README.md
```

## 참고 사항
- RAG tool은 회의록 데이터에서 유사한 Q&A를 검색합니다.
- websearch tool은 실시간 외부 검색을 통해 답변을 보완합니다.
- Langfuse를 통한 LLM 호출 모니터링 및 트레이싱이 가능합니다.
