# main.py

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_tavily import TavilySearch
search_web = TavilySearch(max_results=2)

load_dotenv()

vectordb = None
langfuse = None
callback_handler = None

# Langfuse 클라이언트 초기화
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    callback_handler = CallbackHandler()

### 데이터 로딩 및 벡터 DB 구축
def load_json_docs(json_path: str):
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".[] | {page_content: .content, metadata: {id: .id, title: .title}}",
        text_content=False
    )
    docs = loader.load()
    for doc in docs:
        if isinstance(doc.page_content, str):
            try:
                doc.page_content = doc.page_content.encode('utf-8').decode('unicode_escape')
            except Exception:
                pass
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
    return vectordb

### 도구 정의
@tool
def search_meeting_data(query: str) -> str:
    """
    스타트업 투자 회의록 데이터를 검색합니다.
    """
    global vectordb
    if vectordb is None:
        return "벡터DB가 아직 초기화되지 않았습니다."
    docs = vectordb.similarity_search(query, k=3)
    if len(docs) > 0:
        return "\n\n".join([doc.page_content for doc in docs])
    return "관련 회의록 데이터를 찾을 수 없습니다."

tools = [search_meeting_data, search_web]
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

# LangGraph ReAct agent 생성
react_agent = create_react_agent(
    model=llm,
    tools=tools,
    # system prompt는 MessagesState에서 첫 메시지로 전달
)

def main():
    global vectordb
    st.set_page_config(page_title="스타트업 투자 Q&A", layout="wide")
    st.title("🚀 스타트업 투자 회의록 Q&A (RAG + LangGraph ReAct Agent + Langfuse)")

    # Langfuse 상태 표시
    if langfuse:
        st.success("✅ Langfuse 모니터링이 활성화되었습니다")
    else:
        st.warning("⚠️ Langfuse 환경변수가 설정되지 않았습니다. 모니터링이 비활성화됩니다.")

    with st.spinner("데이터 로딩 중..."):
        docs = load_json_docs("./data/startup_investment_meeting_data_30.json")
        chunks = split_docs(docs)
        vectordb = build_vectorstore(chunks)

    user_input = st.text_input("질문을 입력하세요:", "이번 라운드 투자에서 가장 중점을 둔 부분은 무엇이었나요")
    if st.button("질문하기") and user_input:
        # Langfuse 콜백 핸들러가 있으면 전달 (LangGraph는 config로 콜백 지원 X, 추후 확장)
        system_message = "You are a helpful AI assistant. 회의록 검색 도구를 적극적으로 활용하세요. 답변에 반드시 출처를 명시하세요."
        messages = [
            HumanMessage(content=system_message),
            HumanMessage(content=user_input)
        ]
        result = react_agent.invoke({"messages": messages}, config={"callbacks": [callback_handler] if callback_handler else {}})
        st.markdown("---")
        st.subheader("🧠 답변")
        st.write(result['messages'][-1].content)

if __name__ == "__main__":
    main()