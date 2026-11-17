import os
import re
import json
import uuid
import tempfile
from datetime import datetime
from operator import itemgetter

import pymupdf
import streamlit as st
import streamlit.components.v1 as components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

pymupdf.set_messages(path=os.devnull)
st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="🎓", layout="wide")

PDF = "AIVLE School 백서.pdf"
HOME = "https://aivle.kt.co.kr/home/main/aiDxMain?mcd=HM02002000"
FAQ = "https://aivle.kt.co.kr/home/brd/faq/main?mcd=MC00000056"
FEEDBACK = "feedback.jsonl"

STAGES = {
    "지원 전": ["비전공자도 따라갈 수 있나요?", "지원 자격이 어떻게 되나요?", "선발 절차는 어떻게 진행되나요?", "AI 개발자와 DX 컨설턴트 트랙은 뭐가 다른가요?"],
    "입교 직후": ["하루 공부 시간은 어느 정도가 적당할까요?", "무엇부터 공부해야 하나요?", "지각을 몇 번 하면 결석이 되나요?", "훈련장려금은 어떻게 받나요?"],
    "프로젝트 중": ["미니프로젝트는 어떻게 준비하나요?", "빅프로젝트 주제는 어떻게 정하나요?", "팀 협업은 어떻게 하면 좋을까요?", "발표 전에 무엇을 점검해야 하나요?"],
    "수료·취업 준비": ["수료 기준이 어떻게 되나요?", "어떤 취업 지원을 받을 수 있나요?", "프로젝트를 포트폴리오로 어떻게 만드나요?", "수료 후 어떤 직무로 진출하나요?"],
}

S_MSG = """당신은 KT 에이블스쿨을 먼저 경험한 '선배 에이블러'이자 AIVLE School 학습 도우미입니다.
에이블스쿨 지원을 고민하는 사람과 교육생이 자신 있게 도전하도록 돕습니다.

1. 반드시 [백서]에 있는 내용에만 근거해 답변하세요. 숫자·날짜·조건은 백서에 적힌 그대로 쓰고, 서로 다른 기준의 숫자를 더하거나 새로 계산하지 마세요.
2. [백서]에 질문의 답이 없으면 추측하지 말고 "백서에서 확인되지 않는 내용입니다."로 시작해 한두 문장으로만 답하세요.
3. 핵심부터 간결하게 답하고, 필요하면 번호 목록을 쓰세요. 말투는 친근한 선배의 존댓말(~해요)입니다.
4. 2번이 아니라면 마지막 줄에 백서 내용에 근거해 에이블스쿨에 적극적으로 도전하고 싶어지는 응원 한 문장을 덧붙이세요.

[백서]
{context}"""

PALETTE = {
    True: dict(bg="#0B1210", sf="#111C19", sf2="#12302A", bd="#1F2E2A", tx="#E7F0EC", mu="#9BB0A8", ac="#4FD1A5", ac2="#86E2C2", ub="#1A3A33", sb="#0E1714"),
    False: dict(bg="#FFFFFF", sf="#F4F8F6", sf2="#E3F5EE", bd="#D5E3DD", tx="#111111", mu="#4B5B55", ac="#4FD1A5", ac2="#0F8F6E", ub="#E3F5EE", sb="#F7FAF9"),
}


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_pipeline(filepath, collection_name="aivle_docs", persist_directory="chroma_db", rebuild=True):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(collection_name=collection_name, embedding_function=embedding, persist_directory=persist_directory)
    if rebuild or vectorstore._collection.count() == 0:
        docs = PyMuPDF4LLMLoader(filepath).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""])
        chunks = splitter.split_documents(docs)
        vectorstore.reset_collection()
        _ = vectorstore.add_documents(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([("system", S_MSG), MessagesPlaceholder("history"), ("human", "{input}")])
    rag_chain = (
        RunnablePassthrough.assign(docs=itemgetter("query") | retriever)
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | {"docs": itemgetter("docs"), "answer": prompt | llm | StrOutputParser()}
    )
    return rag_chain


@st.cache_resource(show_spinner="백서를 준비하는 중...")
def default_chain():
    return rag_pipeline(PDF, rebuild=False)


@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


@st.cache_data
def read_pdf(path):
    with open(path, "rb") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def page_info(data):
    doc = pymupdf.open(stream=data, filetype="pdf")
    labels, heads, toc = [], [], set()
    for i, page in enumerate(doc):
        text = page.get_text()
        m = re.search(r"\|\s*(\d+)\s*$", text.strip())
        labels.append(m.group(1) if m else str(i + 1))
        found = [(i, l.strip()) for l in text.splitlines() if re.match(r"^\d{1,2}\.\d{1,2}\s+\S", l.strip())]
        if len(found) < 8:
            heads += found
        else:
            toc.add(i)
    return labels, heads, toc


@st.cache_data(show_spinner=False)
def render_page(data, i, marks=()):
    page = pymupdf.open(stream=data, filetype="pdf")[i]
    for s in marks:
        for r in page.search_for(s):
            a = page.add_highlight_annot(r)
            a.set_colors(stroke=(0.31, 0.82, 0.65))
            a.update()
    return page.get_pixmap(dpi=110).tobytes("png")


def clean(text):
    text = re.sub(r"\|?-{3,}\|?|<br>|\*\*|#+ ", " ", text)
    text = re.sub(r"\|", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def evidence(docs):
    labels, heads, toc = page_info(st.session_state.doc_bytes)
    out = []
    for d in docs:
        i = d.metadata.get("page", 0)
        if i in toc:
            continue
        sec = next((h for p, h in reversed(heads) if p <= i), "")
        out.append({"page": i, "label": labels[i] if i < len(labels) else str(i + 1), "section": sec, "text": clean(d.page_content)})
    return out


def condense(history, q):
    p = ChatPromptTemplate.from_messages([
        ("system", "이전 대화를 참고해 마지막 질문을 혼자 읽어도 이해되는 한 문장의 검색 질문으로 바꾸세요. 질문만 출력하세요."),
        MessagesPlaceholder("history"), ("human", "{input}")])
    return (p | get_llm() | StrOutputParser()).invoke({"history": history, "input": q}).strip()


def followups(q, a, docs):
    p = ChatPromptTemplate.from_messages([
        ("system", "아래 [백서]로 답할 수 있으면서 방금 질문에 이어 궁금해할 만한 짧은 질문 2개를 한 줄에 하나씩, 번호 없이 출력하세요.\n\n[백서]\n{context}"),
        ("human", "질문: {q}\n답변: {a}")])
    out = (p | get_llm() | StrOutputParser()).invoke({"context": format_docs(docs), "q": q, "a": a})
    return [re.sub(r"^[\-\d\.\)\s•]+", "", l).strip() for l in out.splitlines() if l.strip()][:2]


def init():
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("pins", [])
    ss.setdefault("stage", "입교 직후")
    ss.setdefault("view", "chat")
    ss.setdefault("dark", True)
    if "doc_bytes" not in ss:
        ss.doc_bytes, ss.doc_name, ss.doc_id, ss.chain = read_pdf(PDF), PDF, None, None


def ask(q):
    st.session_state.pending = q
    st.session_state.view = "chat"


def goto(mid):
    st.session_state.view = "chat"
    st.session_state.scroll = f"msg-{mid}"


def open_lib(e):
    st.session_state.view = "library"
    st.session_state.lib_target = e


def toggle_pin(mid):
    pins = st.session_state.pins
    pins.remove(mid) if mid in pins else pins.append(mid)


def new_chat():
    st.session_state.messages, st.session_state.pins, st.session_state.view = [], [], "chat"


def reset_doc():
    ss = st.session_state
    ss.doc_bytes, ss.doc_name, ss.doc_id, ss.chain = read_pdf(PDF), PDF, None, None


def save_fb(mid):
    v = st.session_state.get(f"fb_{mid}")
    msgs = st.session_state.messages
    i = next(k for k, m in enumerate(msgs) if m["id"] == mid)
    row = {"time": datetime.now().strftime("%m-%d %H:%M"), "value": v, "q": msgs[i - 1]["content"] if i else "", "a": msgs[i]["content"]}
    with open(FEEDBACK, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scroll_to(anchor, flash=False):
    fx = "const m=e.closest('[data-testid=\"stChatMessage\"]')||e.parentElement;m.animate([{boxShadow:'0 0 0 2px #4FD1A5'},{boxShadow:'0 0 0 2px transparent'}],{duration:2000});" if flash else ""
    components.html(f"""<script>/*{uuid.uuid4()}*/
const d=window.parent.document;let n=0;
const go=()=>{{const e=d.getElementById("{anchor}");if(!e){{if(n++<30)setTimeout(go,150);return;}}
e.scrollIntoView({{behavior:"smooth",block:"start"}});{fx}}};setTimeout(go,300);</script>""", height=0)


def export_text(md):
    lines = []
    for m in st.session_state.messages:
        who = "질문" if m["role"] == "user" else "선배 에이블러"
        lines.append(f"### {who}\n\n{m['content']}\n" if md else f"[{who}]\n{m['content']}\n")
    return "\n".join(lines)


def css(dark):
    c = PALETTE[dark]
    stage_labels = "".join(f'.st-key-stage_{i} button::before{{content:"STEP {i}";display:block;font-size:12px;color:var(--mu)}}' for i in range(4))
    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
:root{{--bg:{c['bg']};--sf:{c['sf']};--sf2:{c['sf2']};--bd:{c['bd']};--tx:{c['tx']};--mu:{c['mu']};--ac:{c['ac']};--ac2:{c['ac2']};--ub:{c['ub']};--sb:{c['sb']}}}
html,body,.stApp,[data-testid="stHeader"],[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{{background:var(--bg)!important;color:var(--tx)}}
.stApp,.stApp p,.stApp li,.stApp label,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp td,.stApp th,.stMarkdown{{color:var(--tx);font-family:'Noto Sans KR',sans-serif}}
[data-testid="stSidebar"]{{background:var(--sb)!important;border-right:1px solid var(--bd)}}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{{color:var(--mu)!important}}
.block-container{{max-width:980px;padding-top:2.2rem}}
.stButton button,[data-testid="stPopover"] button,.stDownloadButton button,.stLinkButton a{{background:var(--sf)!important;color:var(--tx)!important;border:1px solid var(--bd)!important;border-radius:10px}}
.stButton button:hover,.stLinkButton a:hover{{border-color:var(--ac)!important;color:var(--ac2)!important}}
[data-testid="stTextInput"] input,[data-testid="stChatInput"],[data-testid="stChatInput"] textarea,[data-testid="stFileUploaderDropzone"],[data-testid="stExpander"] details,[data-testid="stPopoverBody"]{{background:var(--sf)!important;color:var(--tx)!important;border-color:var(--bd)!important}}
[data-testid="stChatInput"] *,[data-testid="stTextInput"] div[data-baseweb="input"],[data-testid="stTextInput"] div[data-baseweb="input"] *{{background:var(--sf)!important;color:var(--tx)!important}}
[data-testid="stChatInput"] textarea::placeholder,[data-testid="stTextInput"] input::placeholder{{color:var(--mu)!important;opacity:1}}
[data-testid*="stChatMessageAvatar"]{{background:var(--sf2)!important}}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary p{{color:var(--tx)!important}}
[data-testid="stSidebar"] .stButton button,[data-testid="stSidebar"] .stButton button>div,[class*="st-key-stage_"] button>div{{justify-content:flex-start!important;text-align:left;background:transparent!important;border-color:transparent!important;font-size:15px}}
[data-testid="stSidebar"] .stButton button:hover{{background:var(--sf)!important}}
.st-key-nav_lib_on button{{background:var(--sf2)!important;color:var(--ac2)!important}}
.st-key-sresults .stButton button,.st-key-pinlist .stButton button{{font-size:13px;padding:4px 8px;min-height:30px;color:var(--mu)!important}}
[class*="st-key-stage_"] button{{display:block;text-align:left;width:100%;min-height:62px;border-radius:12px!important;font-weight:700}}
[class*="st-key-stage_"] button[kind="primary"]{{background:var(--sf2)!important;border-color:var(--ac)!important}}
[class*="st-key-stage_"] button[kind="primary"]::before{{color:var(--ac)!important}}
{stage_labels}
.st-key-chips button,.st-key-fu button{{border-radius:999px!important;background:transparent!important;border:1px solid var(--ac)!important;color:var(--ac2)!important;font-size:14px}}
.st-key-fu button{{border-color:var(--bd)!important;color:var(--tx)!important;font-size:13px}}
[data-testid="stChatMessage"]{{background:transparent!important}}
[data-testid="stChatMessage"]:has(.u-mark){{flex-direction:row-reverse}}
[data-testid="stChatMessage"]:has(.u-mark) [data-testid="stChatMessageContent"]{{background:var(--ub);border-radius:16px 16px 4px 16px;padding:10px 16px;flex:0 1 auto;max-width:70%}}
[data-testid="stChatMessage"]:has(.a-mark) [data-testid="stChatMessageContent"]{{background:var(--sf);border:1px solid var(--bd);border-radius:4px 16px 16px 16px;padding:12px 18px}}
[class*="st-key-pin_"] button,[class*="st-key-pinon_"] button{{width:30px;height:30px;min-height:30px;padding:0;border-radius:50%!important;font-size:13px}}
[class*="st-key-pin_"] button:hover,[class*="st-key-pinon_"] button{{background:#E5484D!important;border-color:#E5484D!important}}
[class*="st-key-ev_"] button{{font-size:13px;min-height:30px;padding:2px 12px;border-radius:999px!important;border:1px solid var(--ac)!important;color:var(--ac2)!important}}
[class*="st-key-fbw_"]{{opacity:.55;transform:scale(.85)}}
.st-key-darkwrap{{align-items:flex-end}}
.st-key-admin{{opacity:.5}}
.st-key-admin p,.st-key-admin summary p{{font-size:12px!important}}
.ref-box{{border:1px solid var(--ac);background:var(--sf2);border-radius:12px;padding:12px 16px;margin:6px 0 10px;font-size:14px}}
.hint{{text-align:center;font-size:12px;color:var(--mu);margin-top:4px}}
</style>""", unsafe_allow_html=True)


def sidebar():
    ss = st.session_state
    st.logo("assets/aivle_white.png" if ss.dark else "assets/aivle.png", size="large")
    with st.sidebar:
        st.markdown("**학습 도우미**")
        st.button("＋ 새 대화", key="nav_new", on_click=new_chat, width="stretch")
        term = st.text_input("채팅 검색", placeholder="🔍 채팅 검색", label_visibility="collapsed")
        if term:
            with st.container(key="sresults"):
                hits = [m for m in ss.messages if term in m["content"]]
                for m in hits:
                    k = m["content"].find(term)
                    snip = m["content"][max(0, k - 12):k + 28].replace("\n", " ")
                    st.button(f"{'🙋' if m['role'] == 'user' else '🎓'} …{snip}…", key=f"sr_{m['id']}", on_click=goto, args=(m["id"],), width="stretch")
                if not hits:
                    st.caption("검색 결과가 없어요")
        with st.expander(f"📌 저장 ({len(ss.pins)})"):
            with st.container(key="pinlist"):
                for i, m in enumerate(ss.messages):
                    if m["id"] in ss.pins:
                        q = ss.messages[i - 1]["content"] if i else m["content"]
                        st.button(q[:30], key=f"pl_{m['id']}", on_click=goto, args=(m["id"],), width="stretch")
                if not ss.pins:
                    st.caption("답변의 📌를 누르면 여기에 모여요")
        lib_on = ss.view == "library"
        if st.button("📚 자료실", key="nav_lib_on" if lib_on else "nav_lib", width="stretch"):
            ss.view = "chat" if lib_on else "library"
            ss.pop("lib_target", None)
            st.rerun()
        with st.expander("📤 문서 업로드"):
            st.caption(f"현재 문서: {ss.doc_name}")
            f = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")
            if f and f.file_id != ss.doc_id:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                with st.spinner("문서를 벡터화하는 중..."):
                    ss.chain = rag_pipeline(tmp.name, collection_name=f"up_{uuid.uuid4().hex[:8]}", persist_directory=None)
                ss.doc_bytes, ss.doc_name, ss.doc_id = f.getvalue(), f.name, f.file_id
                st.rerun()
            if ss.doc_id:
                st.button("기본 백서로 되돌리기", on_click=reset_doc, width="stretch")
        st.divider()
        with st.container(key="admin"):
            with st.expander("관리자 · 답변 피드백"):
                rows = [json.loads(l) for l in open(FEEDBACK, encoding="utf-8")] if os.path.exists(FEEDBACK) else []
                st.caption(f"👍 {sum(r['value'] == 1 for r in rows)} · 👎 {sum(r['value'] == 0 for r in rows)}")
                for r in reversed([r for r in rows if r["value"] == 0][-10:]):
                    st.caption(f"{r['time']} · {r['q'][:40]}")


def header():
    ss = st.session_state
    c1, c2 = st.columns([0.78, 0.22], vertical_alignment="center")
    c1.markdown("## 지금 어느 단계에 있나요?")
    with c2.container(key="darkwrap"):
        st.toggle("🌙 다크모드", key="dark")
    st.caption("단계를 고르면 그 시기에 선배들이 가장 많이 궁금해했던 질문을 보여 드려요.")
    cols = st.columns(4)
    for i, (col, name) in enumerate(zip(cols, STAGES)):
        col.button(name, key=f"stage_{i}", type="primary" if ss.stage == name else "secondary",
                   on_click=lambda n=name: ss.update(stage=n), width="stretch")
    with st.container(key="chips", horizontal=True):
        for k, q in enumerate(STAGES[ss.stage]):
            st.button(q, key=f"chip_{k}", on_click=ask, args=(q,))


def render_msg(m, last):
    if m["role"] == "user":
        with st.chat_message("user", avatar="🙋"):
            st.markdown(f'<span id="msg-{m["id"]}" class="u-mark"></span>{m["content"]}', unsafe_allow_html=True)
        return
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(f'<span id="msg-{m["id"]}" class="a-mark"></span>', unsafe_allow_html=True)
        with st.container(horizontal=True, vertical_alignment="center"):
            pinned = m["id"] in st.session_state.pins
            st.button("📌", key=f"{'pinon' if pinned else 'pin'}_{m['id']}", on_click=toggle_pin, args=(m["id"],), help="저장")
            meta = "선배 에이블러" if m["notfound"] else f"선배 에이블러 · 백서 {len(m['docs'])}곳 참고"
            st.caption(f"**{meta}**")
        st.markdown(m["content"])
        if m["notfound"]:
            with st.container(horizontal=True):
                st.link_button("공식 홈페이지", HOME)
                st.link_button("자주 묻는 질문(FAQ)", FAQ)
        else:
            with st.container(horizontal=True, vertical_alignment="center"):
                with st.container(key=f"ev_{m['id']}", width="content"):
                    with st.popover(f"📄 참고한 백서 {len(m['docs'])}곳"):
                        for k, e in enumerate(m["docs"]):
                            st.markdown(f"**{e['label']}쪽** · {e['section']}")
                            st.caption(e["text"][:240] + ("…" if len(e["text"]) > 240 else ""))
                            st.button("자료실에서 보기 →", key=f"go_{m['id']}_{k}", on_click=open_lib, args=(e,))
                with st.container(key=f"fbw_{m['id']}", width="content"):
                    st.feedback("thumbs", key=f"fb_{m['id']}", on_change=save_fb, args=(m["id"],))
        if last and m.get("followups"):
            st.caption("다음으로 많이 묻는 질문")
            with st.container(key="fu", horizontal=True):
                for k, q in enumerate(m["followups"]):
                    st.button(q, key=f"fu_{k}", on_click=ask, args=(q,))


def run_turn(q, chain):
    ss = st.session_state
    history = [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in ss.messages[-6:]]
    ss.messages.append({"id": uuid.uuid4().hex[:8], "role": "user", "content": q})
    render_msg(ss.messages[-1], False)
    with st.chat_message("assistant", avatar="🎓"):
        status = st.status("질문을 이해하는 중", expanded=True)
        query = condense(history, q) if history else q
        status.write(f"① 질문 이해 · {query}")
        status.update(label="백서에서 관련 내용을 찾는 중")
        box = {"docs": []}

        def gen():
            for ch in chain.stream({"input": q, "query": query, "history": history}):
                if "docs" in ch:
                    box["docs"] = ch["docs"]
                    status.write(f"② 백서 {len(ch['docs'])}곳 검색")
                    status.update(label="답변을 작성하는 중")
                if "answer" in ch:
                    yield ch["answer"]

        answer = st.write_stream(gen())
        notfound = "확인되지 않는" in answer[:40]
        fu = []
        if not notfound:
            status.update(label="이어질 질문을 고르는 중")
            fu = followups(q, answer, box["docs"])
        status.write("③ 답변 완료")
        status.update(label="답변 완료", state="complete", expanded=False)
    ss.messages.append({"id": uuid.uuid4().hex[:8], "role": "assistant", "content": answer,
                        "docs": [] if notfound else evidence(box["docs"]), "followups": fu, "notfound": notfound})
    st.rerun()


def library():
    ss = st.session_state
    data, t = ss.doc_bytes, ss.get("lib_target")
    labels, _, _ = page_info(data)
    c1, c2 = st.columns([0.7, 0.3], vertical_alignment="center")
    c1.markdown(f"## 📚 자료실\n{ss.doc_name} · {len(labels)}쪽")
    c2.download_button("PDF 내려받기", data, file_name=ss.doc_name, mime="application/pdf", width="stretch")
    if st.button("← 채팅으로 돌아가기"):
        ss.view = "chat"
        st.rerun()
    for i in range(len(labels)):
        st.markdown(f'<div id="pg-{i}"></div>', unsafe_allow_html=True)
        if t and t["page"] == i:
            st.markdown(f'<div class="ref-box"><b>이 답변이 참고한 부분 · {t["label"]}쪽 {t["section"]}</b><br>{t["text"]}</div>', unsafe_allow_html=True)
            marks = tuple(s.strip() for s in re.split(r"(?<=[.:])\s|\n", t["text"]) if len(s.strip()) >= 6)[:15]
            st.image(render_page(data, i, marks), width="stretch")
        else:
            st.image(render_page(data, i), width="stretch")
    if t:
        scroll_to(f"pg-{t['page']}")


init()
ss = st.session_state
css(ss.dark)
sidebar()

if ss.view == "library":
    library()
else:
    header()
    st.divider()
    if not ss.messages:
        with st.chat_message("assistant", avatar="🎓"):
            st.markdown('<span class="a-mark"></span>**선배 에이블러**예요. 위에서 지금 단계를 고르거나 아래에 편하게 물어보세요. 백서에 있는 내용으로만 답해 드려요.', unsafe_allow_html=True)
    for k, m in enumerate(ss.messages):
        render_msg(m, k == len(ss.messages) - 1)
    with st.bottom:
        c1, c2 = st.columns([0.07, 0.93], vertical_alignment="center")
        with c1.popover("⤓", help="대화 내보내기", disabled=not ss.messages):
            st.download_button("Markdown (.md)", export_text(True), file_name="aivle_chat.md")
            st.download_button("텍스트 (.txt)", export_text(False), file_name="aivle_chat.txt")
        typed = c2.chat_input("선배에게 묻듯이 편하게 물어보세요")
        st.markdown('<div class="hint">백서에 없는 내용은 \'확인되지 않음\'으로 답해요</div>', unsafe_allow_html=True)
    q = ss.pop("pending", None) or typed
    if q:
        run_turn(q, ss.chain or default_chain())
    if "scroll" in ss:
        scroll_to(ss.pop("scroll"), flash=True)
