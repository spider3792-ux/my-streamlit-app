

import os, json, smtplib
import pandas as pd
import streamlit as st
from openai import OpenAI
from email.mime.text import MIMEText

# ── 기본 설정 ─────────────────────────────────────
# 키는 환경변수에서 자동
client = OpenAI()

# 이메일 요청 시 받을 주소 (음성 인식 오류를 피하려고 코드에 고정)
MY_EMAIL = "spider379@naver.com"

# 에이블 역할
sys_role = """
당신은 '김 대리'의 개인 비서 '에이블'입니다.
김 대리가 회의에 참석하지 못했을 때, 회의 내용을 정리해 업무를 놓치지 않도록 돕습니다.

[주요 업무]
1. 회의록을 분석하고 핵심 내용을 요약합니다.
2. 회의록에서 김 대리에게 배정된 업무 지시사항을 찾아 자세히 정리합니다.
3. 김 대리의 다양한 질문에 답변합니다.

[답변 규칙]
- 한국어로 정중하고 간결하게 답변합니다.
- 회의록에 없는 내용은 지어내지 않습니다.
- 요약을 주제 → 논의 내용 → 결정 사항 순서로 구역을 나눠서 진행합니다.
- 김 대리 담당 업무는 요약과 별도로 정리하고, 업무마다 업무명, 세부 내용, 기한을 함께 적습니다. 기한이 언급되지 않은 업무는 '기한 미정'으로 표시합니다.
- 결정 사항에는 예산, 일정, 채널처럼 구체적인 수치와 항목을 포함합니다.
- 업무 세부 내용에는 회의에서 언급된 방향과 조건을 포함합니다.
- 기한은 회의에서 해당 업무에 대해 명시적으로 언급된 경우에만 적습니다. 추측하지 않습니다.
- 이메일은 김 대리가 요청한 경우에만 보내고 이메일 본문은 마크다운 기호 없이 작성합니다.
"""

# ── 이메일 발송 함수 (미션 1) ──
def send_email(to, subject, body):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")

    # Secrets에 값이 없으면 연결 전에 바로 안내
    if not sender or not password:
        return {"status": "failed",
                "error": "GMAIL_ADDRESS 또는 GMAIL_APP_PASSWORD가 설정되지 않았습니다."}

    msg = MIMEText(body, "plain", "utf-8")   # 한글 본문 깨짐 방지
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    try:
        # 465 포트: 처음부터 SSL로 연결 (클라우드 서버에서 더 안정적)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as s:
            s.login(sender, password)
            s.sendmail(sender, to, msg.as_string())
        return {"status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

# ── 도구 정의 (미션 7)  ──
# description: LLM이 도구를 언제·어떻게 쓸지 판단하는 기준
tools = [{
    "type": "function",
    "name": "send_email",
    "description": "이메일을 발송합니다. 요청받은 내용은 한 통의 이메일로 모아서 보냅니다.",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "string",
                   "description": "이메일 수신자. 예: 'test@gmail.com'"},
            "subject": {"type": "string",
                        "description": "이메일 제목. 예: '마케팅 전략 초안'"},
            "body": {"type": "string",
                     "description": "이메일 본문. 회의 요약과 김 대리 담당 업무를 모두 포함하고, 마크다운 기호(#, *, -) 없이 일반 텍스트로 작성. 예: '안녕하세요, 김 대리님. 비서 에이블입니다.'"}
        },
        "required": ["to", "subject", "body"],
        "additionalProperties": False
    },
    "strict": True
}]

# ── 업무 우선순위 분석용 출력 스키마 (05 실습 JSON 구조화) ──
priority_schema = {
    "type": "json_schema",
    "name": "priority",
    "schema": {
        "type": "object",
        "properties": {
            "업무목록": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "업무명": {"type": "string"},
                        "기한": {"type": ["string", "null"]},
                        "긴급도": {"type": "string", "enum": ["높음", "낮음"]},
                        "중요도": {"type": "string", "enum": ["높음", "낮음"]},
                        "근거": {"type": "string"}
                    },
                    "required": ["업무명", "기한", "긴급도", "중요도", "근거"],
                    "additionalProperties": False
                }
            },
            "참석자별업무": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "이름": {"type": "string"},
                        "업무수": {"type": "integer"}
                    },
                    "required": ["이름", "업무수"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["업무목록", "참석자별업무"],
        "additionalProperties": False
    }
}

# 우선순위 분석 프롬프트 템플릿
priority_template = """
다음 회의록을 분석하세요.

1. 김 대리에게 배정된 업무를 모두 추출하세요.
   - 기한은 회의록에서 해당 업무에 명시된 표현 그대로 적고, 없으면 null로 하세요.
   - 긴급도: 기한이 가깝거나 회의에서 서둘러야 한다고 언급되면 '높음', 아니면 '낮음'
   - 중요도: 신제품 출시와 매출에 직접 영향을 주면 '높음', 보조 업무면 '낮음'
   - 근거: 판단 이유를 회의록 내용에 근거해 한 문장으로 쓰세요.
2. 회의 참석자(김 대리 포함)별로 새로 배정된 업무 수를 세세요.
   업무를 전달하거나 회의록을 정리하는 일은 업무로 세지 않습니다.

회의록에 없는 내용은 지어내지 마세요.

[회의록]
{meeting}
"""

# 긴급도 × 중요도 → 분류 (파이썬으로 정해서 결과 일관되게)
def classify(urgent, important):
    if urgent == "높음" and important == "높음":
        return "즉시 착수"
    elif important == "높음":
        return "순차 진행"
    else:
        return "위임 검토"

# ── 화면: 제목 ──
st.set_page_config(page_title="AI 개인비서 에이블", page_icon="🤖")
st.title("🤖 AI 개인비서 에이블")
st.caption("음성으로 질문하시면 답변해 드립니다.")

# ── 세션 상태 초기화 (S04: 처음 한 번만) ──
if "messages" not in st.session_state:
    st.session_state.messages = []        # 대화 히스토리
if "meeting_text" not in st.session_state:
    st.session_state.meeting_text = ""    # 업로드한 회의록 (탭 전환 시 유지)

# ── 화면: 회의록 업로드  ──
uploaded_file = st.file_uploader("회의록 파일 업로드 (txt)", type=["txt"])

# 새 파일이 올라오면 세션 상태에 저장 → 다른 탭에 갔다 와도 유지됨
if uploaded_file is not None:
    st.session_state.meeting_text = uploaded_file.read().decode("utf-8")

# 아래 코드는 모두 meeting_text 하나로 사용
meeting_text = st.session_state.meeting_text

if meeting_text == "":
    st.info("회의록 파일을 업로드하면 회의 내용을 바탕으로 답변합니다.")
else:
    st.success("회의록이 준비되었습니다.")
    st.text_area("회의록 내용", meeting_text, height=150)

# ── 화면: 음성 녹음 + 실행 버튼 ──
audio_value = st.audio_input("무엇을 도와드릴까요?")

if st.button("실행"):
    if audio_value is None:
        st.warning("먼저 마이크 버튼을 눌러 질문을 녹음해 주세요.")
    else:
        try:
            # 시간이 걸리는 작업 동안 스피너 표시
            with st.spinner("답변을 준비하고 있습니다.."):

                # STT: 녹음은 파일이 아니라 메모리 데이터 → (파일명, 바이트)로 전달
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("recording.wav", audio_value.getvalue())
                )

                # 프롬프트: 음성 요청 + 고정 이메일 주소 + 회의록
                prompt = f"""
{transcript.text}

[김 대리 이메일 주소]
{MY_EMAIL}

[미팅 내용]
{meeting_text}
"""

                # 1단계: 질문 + 도구 정의 전달
                # auto: 요청 시에만 도구 사용 / parallel False: 메일 한 통만
                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=[{"role": "system", "content": sys_role},
                           {"role": "user", "content": prompt}],
                    tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    temperature=0
                )

                # 도구 호출 요청이 있는지 확인
                tool_calls = [item for item in response.output
                              if item.type == "function_call"]

                if not tool_calls:
                    answer = response.output_text
                else:
                    # 2단계: LLM이 정한 인자(to, subject, body)로 함수 실행
                    tool_outputs = []
                    for item in tool_calls:
                        if item.name == "send_email":
                            args = json.loads(item.arguments)
                            result = send_email(**args)
                            tool_outputs.append({
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": json.dumps(result, ensure_ascii=False)
                            })

                    # 3단계: 실행 결과를 LLM에 돌려주고 최종 답변 받기
                    final_response = client.responses.create(
                        model="gpt-4o-mini",
                        previous_response_id=response.id,
                        input=tool_outputs,
                        temperature=0
                    )
                    answer = final_response.output_text

            # ── 결과 화면 ──────────────────────────
            st.markdown("#### 🎙️ 인식된 질문")
            st.info(transcript.text)   # 음성 인식이 정확한지 확인용

            if tool_calls:   # 메일을 보냈을 때만 발송 결과 표시
                if result["status"] == "success":
                    st.success(f"📧 {args['to']}(으)로 이메일을 보냈습니다.")
                else:
                    st.error(f"이메일 발송 실패: {result.get('error')}")

            st.divider()
            st.markdown("### AI 답변")
            st.markdown(answer)   # 마크다운(###, **)을 서식으로 변환해 표시

            # 대화 히스토리 저장 (질문·답변 한 쌍씩, 최근 10개만 유지)
            st.session_state.messages.append({"role": "user", "content": transcript.text})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.messages = st.session_state.messages[-10:]

        except Exception as e:
            # API·네트워크 오류 시 앱이 멈추지 않고 안내만 표시 (종합실습6)
            st.error(f"오류가 발생했습니다: {e}")

# ── 화면: 대화 기록 (종합실습6 st.chat_message) ──
if st.session_state.messages:
    with st.expander(f"💬 대화 기록 ({len(st.session_state.messages) // 2}건)"):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# 추가 기능: 업무 우선순위 분석 (아이젠하워 매트릭스)

st.divider()
st.subheader("📊 업무 우선순위 분석")
st.caption("업로드한 회의록에서 김 대리 업무를 긴급도 × 중요도로 분류합니다.")

if st.button("우선순위 분석"):
    if meeting_text == "":
        st.warning("먼저 회의록 파일을 업로드해 주세요.")
    else:
        try:
            with st.spinner("업무를 분석하고 있습니다.."):
                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=[{"role": "user",
                            "content": priority_template.format(meeting=meeting_text)}],
                    text={"format": priority_schema},   # 정해진 JSON 형식으로만 답변
                    temperature=0
                )
                data = json.loads(response.output_text)   # JSON 문자열 → 딕셔너리

            # ── ① 업무 집중도 (R&R 밸런스) ──────────
            df_people = pd.DataFrame(data["참석자별업무"])
            kim = df_people[df_people["이름"].str.contains("김")]["업무수"].sum()
            total = df_people["업무수"].sum()

            st.markdown("#### ⚖️ 업무 집중도")
            col1, col2 = st.columns(2)
            col1.metric("김 대리 배정 업무", f"{kim}건")
            col2.metric("전체 배정 업무 중 비중",
                        f"{kim / total * 100:.0f}%" if total else "-")

            # 김 대리 업무가 3건 이상이거나 절반 이상이면 경보
            if kim >= 3 or (total and kim / total >= 0.5):
                st.error("🚨 김 대리에게 업무가 집중되어 있습니다. 우선순위와 위임을 검토하세요.")
            else:
                st.success("업무 배분이 적정한 수준입니다.")

            st.bar_chart(df_people.set_index("이름")["업무수"])

            # ── ② 우선순위 매트릭스 ──────────────────
            df_task = pd.DataFrame(data["업무목록"])
            df_task["기한"] = df_task["기한"].fillna("기한 미정").replace("null", "기한 미정")
            df_task["분류"] = df_task.apply(
                lambda r: classify(r["긴급도"], r["중요도"]), axis=1)

            st.markdown("#### 🗂️ 우선순위 가이드")
            cols = st.columns(3)
            for col, group, icon in zip(cols,
                                        ["즉시 착수", "순차 진행", "위임 검토"],
                                        ["🔴", "🟡", "🔵"]):
                col.markdown(f"**{icon} {group}**")
                tasks = df_task[df_task["분류"] == group]
                if tasks.empty:
                    col.caption("해당 업무 없음")
                for _, t in tasks.iterrows():
                    col.info(f"**{t['업무명']}**\n\n⏰ {t['기한']}")

            # ── ③ 판단 근거 표 ───────────────────────
            st.markdown("#### 📋 판단 근거")
            st.dataframe(df_task[["분류", "업무명", "기한", "긴급도", "중요도", "근거"]],
                         hide_index=True)
            st.caption("※ 중요도는 회의 내용을 바탕으로 한 에이블의 판단입니다.")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

# ══════════════════════════════════════════════════════════════
# 추가 기능 2: 마감 기한 역산 캘린더 (Action Scheduler)
#  - 이 셀은 app.py 끝에 "이어 붙이기"(-a) 합니다.
#  - 반드시 위의 app.py 셀(%%writefile app.py)을 먼저 실행한 뒤, 이 셀을 한 번만 실행하세요.
# ══════════════════════════════════════════════════════════════
from datetime import date, timedelta

# ── ① 일정 추출용 출력 스키마 (05 실습 JSON 구조화) ──
schedule_schema = {
    "type": "json_schema",
    "name": "schedule",
    "schema": {
        "type": "object",
        "properties": {
            "김대리업무": {                      # 김 대리 업무 목록 (기한 없어도 포함)
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "일정명": {"type": "string"},
                        "기한표현": {"type": ["string", "null"]},
                        "기한일수": {"type": ["integer", "null"]}
                    },
                    "required": ["일정명", "기한표현", "기한일수"],
                    "additionalProperties": False
                }
            },
            "주요일정": {                        # 출시일 같은 전체 일정
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "일정명": {"type": "string"},
                        "기한표현": {"type": ["string", "null"]},
                        "기한일수": {"type": ["integer", "null"]}
                    },
                    "required": ["일정명", "기한표현", "기한일수"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["김대리업무", "주요일정"],
        "additionalProperties": False
    }
}

schedule_template = """
다음 회의록에서 일정을 추출하세요.

[김대리업무]
- 김 대리에게 배정된 업무를 업무 하나당 한 항목씩 빠짐없이 모두 적으세요.
- 기한이 없는 업무도 반드시 포함하세요. (기한표현, 기한일수는 null)
- 일정명은 회의록의 구체적인 업무명으로 쓰세요.

[주요일정]
- 신제품 출시일처럼 회의에서 시점이 정해진 전체 일정만 적으세요.
- 다음 회의, 진행 상황 점검처럼 회의 운영 일정은 제외하세요.
- 김대리업무에 적은 업무는 다시 적지 마세요.

[기한 규칙]
- 기한표현: 회의록에 나온 시한 표현을 그대로 적고, 없으면 null로 하세요.
- 기한일수: 회의 날로부터 며칠 뒤인지 정수로 적으세요.
  '다음 주까지'는 7, '2주 안에'는 14처럼 회의 날 기준으로 정할 수 있을 때만 적습니다.
  '다음 달 초'처럼 정하기 어렵거나, '출시 2주 전'처럼 다른 일정을 기준으로 한 시점이거나,
  기한이 없으면 null로 하세요.

회의록에 없는 내용은 지어내지 마세요.

[회의록]
{meeting}
"""

# ── ② 날짜 계산 (계산은 파이썬이 담당 → 결과가 항상 일관됨) ──
def calc_due(base, expr, days):
    """기준일 + 기한일수로 마감일 계산. '다음 달'이면 다음 달 1일. 못 정하면 None"""
    if days is not None:
        return base + timedelta(days=days)
    if expr and "다음 달" in expr:
        return (base.replace(day=1) + timedelta(days=32)).replace(day=1)
    return None

def make_milestones(name, due, base):
    """마감일 기준 D-14, D-7, D-1, D-day 생성 (기준일보다 이전 날짜는 제외)"""
    rows = []
    for label, before in [("D-14", 14), ("D-7", 7), ("D-1", 1), ("D-day", 0)]:
        day = due - timedelta(days=before)
        if day >= base:
            rows.append({"날짜": day, "단계": label, "일정명": name,
                         "제목": f"[{label}] {name}"})
    return rows

# ── ③ 캘린더 파일(.ics) 만들기 → 어떤 캘린더에서도 열면 바로 등록 ──
def make_ics(rows):
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Able//Action Scheduler//KO"]
    for i, r in enumerate(rows):
        lines += ["BEGIN:VEVENT",
                  f"UID:able-{r['날짜']:%Y%m%d}-{i}@able",
                  f"DTSTAMP:{date.today():%Y%m%d}T000000Z",
                  f"DTSTART;VALUE=DATE:{r['날짜']:%Y%m%d}",
                  f"DTEND;VALUE=DATE:{r['날짜'] + timedelta(days=1):%Y%m%d}",
                  f"SUMMARY:{r['제목']}",
                  "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)

# ── ④ Google 캘린더 연결 (서비스 계정 + 내 캘린더 공유) ──
# 환경변수 GOOGLE_TOKEN: service_account.json 내용 / GOOGLE_CALENDAR_ID: 내 캘린더 ID
#   - Colab: api_key.txt 셀에서 등록
#   - Streamlit Cloud: Secrets에 등록
def get_calendar_service():
    from googleapiclient.discovery import build
    info = json.loads(os.getenv("GOOGLE_TOKEN"))
    if info.get("type") == "service_account":      # 서비스 계정 키를 쓴 경우
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/calendar"])
    else:                                           # OAuth로 받은 token.json
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_info(info)
    return build("calendar", "v3", credentials=creds)

def add_to_google(rows):
    service = get_calendar_service()
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")   # 기본: 본인 캘린더
    for r in rows:
        event = {
            "summary": r["제목"],
            "description": "AI 개인비서 에이블이 회의록 기한을 역산해 등록한 일정입니다.",
            "start": {"date": r["날짜"].isoformat()},
            "end": {"date": (r["날짜"] + timedelta(days=1)).isoformat()},   # 종일 일정
        }
        service.events().insert(calendarId=calendar_id, body=event).execute()
    return len(rows)

# ── ⑤ 화면 ───────────────────────────────────────
st.divider()
st.subheader("📅 마감 기한 역산 캘린더")
st.caption("회의록의 시한을 날짜로 계산해 D-14, D-7, D-1 마일스톤으로 등록합니다.")

if "schedule" not in st.session_state:
    st.session_state.schedule = []      # 추출한 일정 (버튼을 눌러도 유지)

base_day = st.date_input("회의 날짜 (기한 계산 기준일)", value=date.today())

if st.button("일정 추출"):
    if meeting_text == "":
        st.warning("먼저 회의록 파일을 업로드해 주세요.")
    else:
        try:
            with st.spinner("회의록에서 일정을 찾고 있습니다.."):
                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=[{"role": "user",
                            "content": schedule_template.format(meeting=meeting_text)}],
                    text={"format": schedule_schema},
                    temperature=0
                )
                data = json.loads(response.output_text)

            # 이전에 수정했던 체크박스·날짜 값 지우기 (새 결과로 다시 채우기 위해)
            for key in list(st.session_state.keys()):
                if key.startswith(("use_", "due_")):
                    del st.session_state[key]

                        # 두 목록을 하나로 합치면서 구분 표시 (김 대리 업무가 먼저 나오게)
            schedule = []
            for item in data["김대리업무"]:
                schedule.append({**item, "구분": "김 대리 업무"})
            for item in data["주요일정"]:
                schedule.append({**item, "구분": "주요 일정"})
            st.session_state.schedule = schedule
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

# 추출 결과가 있으면: 확인·수정 → 마일스톤 → 등록
if st.session_state.schedule:
    st.markdown("#### ✅ 일정 확인 및 수정")
    st.caption("기한이 없거나 애매한 일정은 날짜를 직접 정한 뒤 체크하세요.")

    all_rows = []
    for i, item in enumerate(st.session_state.schedule):
        # 구분 이름만 있는 빈 항목은 건너뛰기 (AI 출력 검증)
        if item["일정명"] in ("김 대리 업무", "주요 일정"):
            continue
        # "null" 글자도 기한 없음으로 처리
        expr = item["기한표현"] if item["기한표현"] not in (None, "", "null") else None

        due = calc_due(base_day, expr, item["기한일수"])
        expr = expr or "기한 미정"

        c1, c2, c3 = st.columns([1, 3, 2])
        use = c1.checkbox("등록", value=due is not None, key=f"use_{i}")
        c2.markdown(f"**{item['일정명']}**  \n{item['구분']} · 회의 표현: {expr}")
        due = c3.date_input("마감일", value=due or base_day + timedelta(days=7),
                            key=f"due_{i}")

        if use:
            all_rows += make_milestones(item["일정명"], due, base_day)

    if all_rows:
        all_rows.sort(key=lambda r: r["날짜"])
        st.markdown("#### 🗓️ 등록될 마일스톤")
        st.dataframe(pd.DataFrame(all_rows)[["날짜", "단계", "일정명"]], hide_index=True)

        col1, col2 = st.columns(2)

        # 방법 1: Google 캘린더 자동 등록 (토큰이 있을 때만 버튼 활성화)
        has_token = bool(os.getenv("GOOGLE_TOKEN"))
        if col1.button("Google 캘린더에 등록", disabled=not has_token):
            try:
                with st.spinner("Google 캘린더에 등록하고 있습니다.."):
                    count = add_to_google(all_rows)
                st.success(f"📅 {count}개 일정을 Google 캘린더에 등록했습니다.")
            except Exception as e:
                st.error(f"캘린더 등록 실패: {e}")
        if not has_token:
            col1.caption("GOOGLE_TOKEN이 없어 비활성화됨")

        # 방법 2: .ics 파일 다운로드 (어떤 캘린더든 열면 등록됨)
        col2.download_button("📥 캘린더 파일(.ics) 다운로드",
                             data=make_ics(all_rows),
                             file_name="able_schedule.ics",
                             mime="text/calendar")
    else:
        st.info("등록할 일정을 체크해 주세요.")

# ══════════════════════════════════════════════════════════════
# 추가 기능 2: 마감 기한 역산 캘린더 (Action Scheduler)
#  - 이 셀은 app.py 끝에 "이어 붙이기"(-a) 합니다.
#  - 반드시 위의 app.py 셀(%%writefile app.py)을 먼저 실행한 뒤, 이 셀을 한 번만 실행하세요.
# ══════════════════════════════════════════════════════════════
from datetime import date, timedelta

# ── ① 일정 추출용 출력 스키마 (05 실습 JSON 구조화) ──
schedule_schema = {
    "type": "json_schema",
    "name": "schedule",
    "schema": {
        "type": "object",
        "properties": {
            "김대리업무": {                      # 김 대리 업무 목록 (기한 없어도 포함)
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "일정명": {"type": "string"},
                        "기한표현": {"type": ["string", "null"]},
                        "기한일수": {"type": ["integer", "null"]}
                    },
                    "required": ["일정명", "기한표현", "기한일수"],
                    "additionalProperties": False
                }
            },
            "주요일정": {                        # 출시일 같은 전체 일정
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "일정명": {"type": "string"},
                        "기한표현": {"type": ["string", "null"]},
                        "기한일수": {"type": ["integer", "null"]}
                    },
                    "required": ["일정명", "기한표현", "기한일수"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["김대리업무", "주요일정"],
        "additionalProperties": False
    }
}

schedule_template = """
다음 회의록에서 일정을 추출하세요.

[김대리업무]
- 김 대리에게 배정된 업무를 업무 하나당 한 항목씩 빠짐없이 모두 적으세요.
- 기한이 없는 업무도 반드시 포함하세요. (기한표현, 기한일수는 null)
- 일정명은 회의록의 구체적인 업무명으로 쓰세요.

[주요일정]
- 신제품 출시일처럼 회의에서 시점이 정해진 전체 일정만 적으세요.
- 다음 회의, 진행 상황 점검처럼 회의 운영 일정은 제외하세요.
- 김대리업무에 적은 업무는 다시 적지 마세요.

[기한 규칙]
- 기한표현: 회의록에 나온 시한 표현을 그대로 적고, 없으면 null로 하세요.
- 기한일수: 회의 날로부터 며칠 뒤인지 정수로 적으세요.
  '다음 주까지'는 7, '2주 안에'는 14처럼 회의 날 기준으로 정할 수 있을 때만 적습니다.
  '다음 달 초'처럼 정하기 어렵거나, '출시 2주 전'처럼 다른 일정을 기준으로 한 시점이거나,
  기한이 없으면 null로 하세요.

회의록에 없는 내용은 지어내지 마세요.

[회의록]
{meeting}
"""

# ── ② 날짜 계산 (계산은 파이썬이 담당 → 결과가 항상 일관됨) ──
def calc_due(base, expr, days):
    """기준일 + 기한일수로 마감일 계산. '다음 달'이면 다음 달 1일. 못 정하면 None"""
    if days is not None:
        return base + timedelta(days=days)
    if expr and "다음 달" in expr:
        return (base.replace(day=1) + timedelta(days=32)).replace(day=1)
    return None

def make_milestones(name, due, base):
    """마감일 기준 D-14, D-7, D-1, D-day 생성 (기준일보다 이전 날짜는 제외)"""
    rows = []
    for label, before in [("D-14", 14), ("D-7", 7), ("D-1", 1), ("D-day", 0)]:
        day = due - timedelta(days=before)
        if day >= base:
            rows.append({"날짜": day, "단계": label, "일정명": name,
                         "제목": f"[{label}] {name}"})
    return rows

# ── ③ 캘린더 파일(.ics) 만들기 → 어떤 캘린더에서도 열면 바로 등록 ──
def make_ics(rows):
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Able//Action Scheduler//KO"]
    for i, r in enumerate(rows):
        lines += ["BEGIN:VEVENT",
                  f"UID:able-{r['날짜']:%Y%m%d}-{i}@able",
                  f"DTSTAMP:{date.today():%Y%m%d}T000000Z",
                  f"DTSTART;VALUE=DATE:{r['날짜']:%Y%m%d}",
                  f"DTEND;VALUE=DATE:{r['날짜'] + timedelta(days=1):%Y%m%d}",
                  f"SUMMARY:{r['제목']}",
                  "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)

# ── ④ Google 캘린더 연결 (서비스 계정 + 내 캘린더 공유) ──
# 환경변수 GOOGLE_TOKEN: service_account.json 내용 / GOOGLE_CALENDAR_ID: 내 캘린더 ID
#   - Colab: api_key.txt 셀에서 등록
#   - Streamlit Cloud: Secrets에 등록
def get_calendar_service():
    from googleapiclient.discovery import build
    info = json.loads(os.getenv("GOOGLE_TOKEN"))
    if info.get("type") == "service_account":      # 서비스 계정 키를 쓴 경우
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/calendar"])
    else:                                           # OAuth로 받은 token.json
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_info(info)
    return build("calendar", "v3", credentials=creds)

def add_to_google(rows):
    service = get_calendar_service()
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")   # 기본: 본인 캘린더
    for r in rows:
        event = {
            "summary": r["제목"],
            "description": "AI 개인비서 에이블이 회의록 기한을 역산해 등록한 일정입니다.",
            "start": {"date": r["날짜"].isoformat()},
            "end": {"date": (r["날짜"] + timedelta(days=1)).isoformat()},   # 종일 일정
        }
        service.events().insert(calendarId=calendar_id, body=event).execute()
    return len(rows)

# ── ⑤ 화면 ───────────────────────────────────────
st.divider()
st.subheader("📅 마감 기한 역산 캘린더")
st.caption("회의록의 시한을 날짜로 계산해 D-14, D-7, D-1 마일스톤으로 등록합니다.")

if "schedule" not in st.session_state:
    st.session_state.schedule = []      # 추출한 일정 (버튼을 눌러도 유지)

base_day = st.date_input("회의 날짜 (기한 계산 기준일)", value=date.today())

if st.button("일정 추출"):
    if meeting_text == "":
        st.warning("먼저 회의록 파일을 업로드해 주세요.")
    else:
        try:
            with st.spinner("회의록에서 일정을 찾고 있습니다.."):
                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=[{"role": "user",
                            "content": schedule_template.format(meeting=meeting_text)}],
                    text={"format": schedule_schema},
                    temperature=0
                )
                data = json.loads(response.output_text)

            # 이전에 수정했던 체크박스·날짜 값 지우기 (새 결과로 다시 채우기 위해)
            for key in list(st.session_state.keys()):
                if key.startswith(("use_", "due_")):
                    del st.session_state[key]

                        # 두 목록을 하나로 합치면서 구분 표시 (김 대리 업무가 먼저 나오게)
            schedule = []
            for item in data["김대리업무"]:
                schedule.append({**item, "구분": "김 대리 업무"})
            for item in data["주요일정"]:
                schedule.append({**item, "구분": "주요 일정"})
            st.session_state.schedule = schedule
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

# 추출 결과가 있으면: 확인·수정 → 마일스톤 → 등록
if st.session_state.schedule:
    st.markdown("#### ✅ 일정 확인 및 수정")
    st.caption("기한이 없거나 애매한 일정은 날짜를 직접 정한 뒤 체크하세요.")

    all_rows = []
    for i, item in enumerate(st.session_state.schedule):
        # 구분 이름만 있는 빈 항목은 건너뛰기 (AI 출력 검증)
        if item["일정명"] in ("김 대리 업무", "주요 일정"):
            continue
        # "null" 글자도 기한 없음으로 처리
        expr = item["기한표현"] if item["기한표현"] not in (None, "", "null") else None

        due = calc_due(base_day, expr, item["기한일수"])
        expr = expr or "기한 미정"

        c1, c2, c3 = st.columns([1, 3, 2])
        use = c1.checkbox("등록", value=due is not None, key=f"use_{i}")
        c2.markdown(f"**{item['일정명']}**  \n{item['구분']} · 회의 표현: {expr}")
        due = c3.date_input("마감일", value=due or base_day + timedelta(days=7),
                            key=f"due_{i}")

        if use:
            all_rows += make_milestones(item["일정명"], due, base_day)

    if all_rows:
        all_rows.sort(key=lambda r: r["날짜"])
        st.markdown("#### 🗓️ 등록될 마일스톤")
        st.dataframe(pd.DataFrame(all_rows)[["날짜", "단계", "일정명"]], hide_index=True)

        col1, col2 = st.columns(2)

        # 방법 1: Google 캘린더 자동 등록 (토큰이 있을 때만 버튼 활성화)
        has_token = bool(os.getenv("GOOGLE_TOKEN"))
        if col1.button("Google 캘린더에 등록", disabled=not has_token):
            try:
                with st.spinner("Google 캘린더에 등록하고 있습니다.."):
                    count = add_to_google(all_rows)
                st.success(f"📅 {count}개 일정을 Google 캘린더에 등록했습니다.")
            except Exception as e:
                st.error(f"캘린더 등록 실패: {e}")
        if not has_token:
            col1.caption("GOOGLE_TOKEN이 없어 비활성화됨")

        # 방법 2: .ics 파일 다운로드 (어떤 캘린더든 열면 등록됨)
        col2.download_button("📥 캘린더 파일(.ics) 다운로드",
                             data=make_ics(all_rows),
                             file_name="able_schedule.ics",
                             mime="text/calendar")
    else:
        st.info("등록할 일정을 체크해 주세요.")
# ── [임시] 메일 연결 진단 (확인 후 삭제) ──
with st.sidebar:
    if st.button("📧 메일 연결 진단"):
        sender = os.getenv("GMAIL_ADDRESS")
        password = os.getenv("GMAIL_APP_PASSWORD")
        st.write("주소 등록:", bool(sender), "/ 비밀번호 길이:", len(password or ""))
        for port in (465, 587):
            try:
                if port == 465:
                    s = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15)
                else:
                    s = smtplib.SMTP("smtp.gmail.com", 587, timeout=15)
                    s.starttls()
                st.write(f"{port} 연결 성공")
                s.login(sender, password)
                st.success(f"{port} 로그인 성공")
                s.quit()
            except Exception as e:
                st.error(f"{port} 실패: {type(e).__name__}: {e}")
