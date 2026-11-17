
import os
os.environ["BOARD_ADMIN_PASSWORD"] = "본인이_정한_관리자_비밀번호"


# ============================================================
# AIVLE SCHOOL - 선배 Q&A / 선배 멘토
# 담당 기능 독립 모듈
# 기존 RAG / Chroma / PDF 업로드 기능과 분리
# ============================================================

import streamlit as st
import sqlite3
import tempfile
import os
import secrets
import hashlib
import hmac
from pathlib import Path
from contextlib import closing

DB_PATH = Path(tempfile.gettempdir()) / "aivle_senior_board.sqlite3"

# ============================================================
# 1. SQLite 데이터베이스
# ============================================================

def db_connect():
    conn = sqlite3.connect(str(DB_PATH), timeout=15)
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA foreign_keys=ON")

    # 기존 질문 게시판 테이블과 호환
    conn.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL
                DEFAULT (datetime('now','localtime'))
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS replies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL,
            author TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL
                DEFAULT (datetime('now','localtime')),
            FOREIGN KEY(question_id) REFERENCES questions(id)
        )
    """)

    # 선배 등록 및 인증
    conn.execute("""
        CREATE TABLE IF NOT EXISTS senior_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            generation TEXT NOT NULL,
            track TEXT NOT NULL,
            company TEXT NOT NULL DEFAULT '',
            job TEXT NOT NULL DEFAULT '',
            introduction TEXT NOT NULL DEFAULT '',
            email TEXT NOT NULL,
            wants_mentor INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            code_hash TEXT,
            created_at TEXT NOT NULL
                DEFAULT (datetime('now','localtime'))
        )
    """)

    # 멘토링 신청
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mentoring_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            senior_id INTEGER NOT NULL,
            junior_name TEXT NOT NULL,
            junior_email TEXT NOT NULL,
            topic TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
                DEFAULT (datetime('now','localtime')),
            FOREIGN KEY(senior_id) REFERENCES senior_profiles(id)
        )
    """)

    conn.commit()
    return conn

def db_query(sql, params=()):
    with closing(db_connect()) as conn:
        return conn.execute(sql, params).fetchall()

def db_execute(sql, params=()):
    with closing(db_connect()) as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.rowcount

# ============================================================
# 2. 선배 인증 코드
# ============================================================

def get_admin_password():
    try:
        value = st.secrets.get("BOARD_ADMIN_PASSWORD", "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get("BOARD_ADMIN_PASSWORD", "")

def hash_code(code):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        code.encode("utf-8"),
        salt,
        200000
    )
    return f"{salt.hex()}:{digest.hex()}"

def check_code(code, stored):
    try:
        salt_hex, digest_hex = stored.split(":")
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            code.encode("utf-8"),
            bytes.fromhex(salt_hex),
            200000
        )
        return hmac.compare_digest(
            digest, bytes.fromhex(digest_hex)
        )
    except (ValueError, AttributeError, TypeError):
        return False

def verify_senior(senior_id, code):
    rows = db_query("""
        SELECT code_hash
        FROM senior_profiles
        WHERE id = ? AND status = 'approved'
    """, (senior_id,))

    if not rows or not rows[0][0]:
        return False

    return check_code(code, rows[0][0])

def valid_email(email):
    email = email.strip()
    return (
        "@" in email
        and "." in email.split("@")[-1]
        and " " not in email
    )

def back_to_chat():
    if st.button(
        "← AI 학습 도우미로 돌아가기",
        key="senior_back_" + st.session_state.app_view
    ):
        st.session_state.app_view = "chat"
        st.rerun()

# ============================================================
# 3. 선배 등록 신청 (공통)
# ============================================================

def render_senior_registration():
    with st.expander("🎓 졸업 선배 등록 / 인증 신청"):
        st.caption(
            "Q&A 답변에 참여하거나 멘토로 등록하려면 "
            "관리자 승인이 필요합니다."
        )

        with st.form(
            "senior_registration",
            clear_on_submit=True
        ):
            name = st.text_input(
                "이름 또는 닉네임",
                max_chars=30
            )

            generation = st.text_input(
                "에이블스쿨 기수",
                placeholder="예: 6기",
                max_chars=20
            )

            track = st.selectbox(
                "교육 트랙",
                ["DX 컨설턴트", "AI 개발자", "기타"]
            )

            email = st.text_input(
                "관리자 확인용 이메일",
                max_chars=200
            )

            wants_mentor = st.checkbox(
                "멘토로도 활동하겠습니다."
            )

            company = st.text_input(
                "재직 회사 (멘토 지원 시 필수)",
                max_chars=100
            )

            job = st.text_input(
                "담당 직무 (멘토 지원 시 필수)",
                max_chars=100
            )

            introduction = st.text_area(
                "자기소개 및 상담 가능한 분야",
                max_chars=1000,
                placeholder=(
                    "예: 데이터 분석가로 근무 중이며 "
                    "포트폴리오와 취업 준비에 도움을 드릴 수 있습니다."
                )
            )

            submit = st.form_submit_button(
                "선배 등록 신청",
                use_container_width=True
            )

        if submit:
            if not name.strip() or not generation.strip():
                st.warning("이름과 기수를 입력해 주세요.")

            elif not valid_email(email):
                st.warning("올바른 이메일을 입력해 주세요.")

            elif wants_mentor and (
                not company.strip() or not job.strip()
            ):
                st.warning(
                    "멘토로 활동하려면 회사와 직무를 입력해야 합니다."
                )

            else:
                db_execute("""
                    INSERT INTO senior_profiles (
                        name, generation, track, company, job,
                        introduction, email, wants_mentor
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    name.strip(),
                    generation.strip(),
                    track,
                    company.strip(),
                    job.strip(),
                    introduction.strip(),
                    email.strip(),
                    int(wants_mentor)
                ))

                st.success(
                    "선배 등록 신청이 완료되었습니다. "
                    "관리자 확인 후 승인됩니다."
                )

# ============================================================
# 4. 선배 Q&A
# ============================================================

def render_senior_qna():
    with st.container(key="senior_board_page"):
        st.markdown("## 💬 선배 Q&A")
        st.write(
            "에이블스쿨 선배에게 궁금한 점을 질문하고 "
            "선배의 답변을 확인해 보세요."
        )
        back_to_chat()
        st.divider()

        # 질문 작성
        with st.expander("✍️ 질문 작성하기", expanded=False):
            with st.form(
                "senior_question_form",
                clear_on_submit=True
            ):
                author = st.text_input(
                    "이름 또는 닉네임",
                    max_chars=30,
                    placeholder="예: 10기 교육생"
                )

                title = st.text_input(
                    "질문 제목",
                    max_chars=120,
                    placeholder="예: DX 프로젝트는 어떻게 준비하셨나요?"
                )

                body = st.text_area(
                    "질문 내용",
                    max_chars=4000,
                    height=140
                )

                submit = st.form_submit_button(
                    "질문 등록",
                    use_container_width=True
                )

            if submit:
                if not title.strip() or not body.strip():
                    st.warning("제목과 내용을 입력해 주세요.")
                else:
                    db_execute("""
                        INSERT INTO questions (author, title, body)
                        VALUES (?, ?, ?)
                    """, (
                        author.strip() or "익명 후배",
                        title.strip(),
                        body.strip()
                    ))
                    st.rerun()

        # 질문 검색
        st.markdown("### 📋 질문 목록")

        search = st.text_input(
            "질문 검색",
            placeholder="제목 또는 키워드를 검색하세요.",
            key="senior_qna_search"
        )

        questions = db_query("""
            SELECT id, author, title, body, created_at
            FROM questions
            ORDER BY id DESC
        """)

        if search.strip():
            keyword = search.strip().lower()
            questions = [
                q for q in questions
                if keyword in (q[2] + " " + q[3]).lower()
            ]

        approved = db_query("""
            SELECT id, name, generation, company
            FROM senior_profiles
            WHERE status = 'approved'
            ORDER BY id DESC
        """)

        senior_labels = {
            sid: f"{name} · {generation} · {company or '졸업 선배'}"
            for sid, name, generation, company in approved
        }

        if not questions:
            st.info("아직 등록된 질문이 없습니다.")

        for qid, author, title, body, created in questions:
            replies = db_query("""
                SELECT author, body, created_at
                FROM replies
                WHERE question_id = ?
                ORDER BY id ASC
            """, (qid,))

            status = (
                "✅ 답변 완료" if replies else "⏳ 답변 대기"
            )

            with st.expander(
                f"{status} | {title} | 답변 {len(replies)}개"
            ):
                st.caption(f"👤 {author} · {created}")
                st.write(body)
                st.divider()

                for reply_author, reply_body, reply_date in replies:
                    st.markdown(f"**🎓 {reply_author} 선배**")
                    st.caption(reply_date)
                    st.write(reply_body)
                    st.divider()

                if not approved:
                    st.caption(
                        "현재 등록된 인증 선배가 없습니다."
                    )
                    continue

                with st.form(
                    f"senior_reply_form_{qid}",
                    clear_on_submit=True
                ):
                    senior_id = st.selectbox(
                        "답변 작성 선배",
                        options=list(senior_labels.keys()),
                        format_func=lambda sid: senior_labels[sid]
                    )

                    code = st.text_input(
                        "선배 인증 코드",
                        type="password"
                    )

                    reply = st.text_area(
                        "선배 답변",
                        max_chars=4000
                    )

                    reply_submit = st.form_submit_button(
                        "답변 등록"
                    )

                if reply_submit:
                    if not reply.strip():
                        st.warning("답변 내용을 입력해 주세요.")

                    elif not verify_senior(senior_id, code):
                        st.error("선배 인증 코드가 올바르지 않습니다.")

                    else:
                        senior_name = db_query("""
                            SELECT name FROM senior_profiles
                            WHERE id = ?
                        """, (senior_id,))[0][0]

                        db_execute("""
                            INSERT INTO replies
                            (question_id, author, body)
                            VALUES (?, ?, ?)
                        """, (
                            qid, senior_name, reply.strip()
                        ))

                        st.rerun()

        st.divider()
        render_senior_registration()

# ============================================================
# 5. 선배 멘토
# ============================================================

def render_senior_mentor():
    with st.container(key="senior_board_page"):
        st.markdown("## 🤝 선배 멘토")
        st.write(
            "에이블스쿨 졸업 후 취업한 선배에게 "
            "진로 및 취업 상담을 신청해 보세요."
        )
        back_to_chat()
        st.divider()

        st.markdown("### 🔎 멘토 찾기")

        col1, col2 = st.columns([1, 2])

        with col1:
            track_filter = st.selectbox(
                "교육 트랙",
                ["전체", "DX 컨설턴트", "AI 개발자", "기타"],
                key="mentor_track_filter"
            )

        with col2:
            keyword = st.text_input(
                "멘토 검색",
                placeholder="회사, 직무, 이름 검색",
                key="mentor_keyword"
            )

        mentors = db_query("""
            SELECT id, name, generation, track,
                   company, job, introduction
            FROM senior_profiles
            WHERE status = 'approved'
              AND wants_mentor = 1
            ORDER BY id DESC
        """)

        if track_filter != "전체":
            mentors = [
                m for m in mentors if m[3] == track_filter
            ]

        if keyword.strip():
            word = keyword.strip().lower()
            mentors = [
                m for m in mentors
                if word in " ".join(
                    str(v or "") for v in m[1:]
                ).lower()
            ]

        if not mentors:
            st.info("현재 조건에 맞는 멘토가 없습니다.")

        else:
            st.caption(f"검색된 멘토 {len(mentors)}명")

        for mid, name, generation, track, company, job, intro in mentors:
            with st.container(border=True):
                st.markdown(f"### 🎓 {name} 선배")
                st.caption(
                    f"✅ 관리자 승인 · {generation} · {track}"
                )

                st.write(f"🏢 **회사:** {company}")
                st.write(f"💼 **직무:** {job}")
                st.write(intro)

                with st.expander("✉️ 멘토링 신청"):
                    with st.form(
                        f"mentoring_form_{mid}",
                        clear_on_submit=True
                    ):
                        junior_name = st.text_input(
                            "신청자 이름 또는 닉네임",
                            max_chars=30
                        )

                        junior_email = st.text_input(
                            "연락받을 이메일",
                            max_chars=200
                        )

                        topic = st.selectbox(
                            "멘토링 주제",
                            [
                                "취업 및 진로",
                                "포트폴리오",
                                "면접 준비",
                                "프로젝트",
                                "직무 상담",
                                "기타"
                            ]
                        )

                        message = st.text_area(
                            "상담하고 싶은 내용",
                            max_chars=2000
                        )

                        submit = st.form_submit_button(
                            "멘토링 신청",
                            use_container_width=True
                        )

                    if submit:
                        if not junior_name.strip() or not message.strip():
                            st.warning("이름과 상담 내용을 입력하세요.")

                        elif not valid_email(junior_email):
                            st.warning("올바른 이메일을 입력하세요.")

                        else:
                            db_execute("""
                                INSERT INTO mentoring_requests (
                                    senior_id, junior_name,
                                    junior_email, topic, message
                                )
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                mid,
                                junior_name.strip(),
                                junior_email.strip(),
                                topic,
                                message.strip()
                            ))

                            st.success(
                                "멘토링 신청이 접수되었습니다. "
                                "운영자가 신청 내용을 확인합니다."
                            )

        st.divider()
        render_senior_registration()

# ============================================================
# 6. 운영자 관리
# ============================================================

def render_senior_admin():
    with st.container(key="senior_board_page"):
        st.markdown("## 🔐 선배 연결 운영자")
        back_to_chat()

        password = get_admin_password()

        if not password:
            st.error(
                "BOARD_ADMIN_PASSWORD 설정이 필요합니다."
            )
            return

        if not st.session_state.get("senior_admin_ok", False):
            with st.form("senior_admin_login"):
                entered = st.text_input(
                    "관리자 비밀번호",
                    type="password"
                )
                login = st.form_submit_button("로그인")

            if login:
                if hmac.compare_digest(entered, password):
                    st.session_state.senior_admin_ok = True
                    st.rerun()
                else:
                    st.error("비밀번호가 일치하지 않습니다.")
            return

        if st.button("관리자 로그아웃"):
            st.session_state.senior_admin_ok = False
            st.session_state.pop("new_senior_code", None)
            st.rerun()

        tab1, tab2 = st.tabs([
            "선배 등록 승인",
            "멘토링 신청 관리"
        ])

        with tab1:
            pending = db_query("""
                SELECT id, name, generation, track,
                       company, job, introduction, email,
                       wants_mentor
                FROM senior_profiles
                WHERE status = 'pending'
                ORDER BY id ASC
            """)

            if not pending:
                st.info("현재 승인 대기 중인 선배가 없습니다.")

            for (
                sid, name, generation, track,
                company, job, intro, email, wants_mentor
            ) in pending:
                with st.container(border=True):
                    st.markdown(f"**{name} · {generation}**")
                    st.write(f"트랙: {track}")
                    st.write(f"활동: {'Q&A + 멘토' if wants_mentor else 'Q&A'}")
                    st.write(f"회사: {company or '미입력'}")
                    st.write(f"직무: {job or '미입력'}")
                    st.write(f"소개: {intro or '미입력'}")
                    st.write(f"확인 이메일: {email}")

                    st.warning(
                        "승인 전 졸업 여부를 별도 경로로 확인하세요. "
                        "멘토 지원자는 재직 여부도 확인해야 합니다."
                    )

                    c1, c2 = st.columns(2)

                    with c1:
                        if st.button(
                            "✅ 승인",
                            key=f"senior_approve_{sid}",
                            use_container_width=True
                        ):
                            code = secrets.token_urlsafe(18)

                            db_execute("""
                                UPDATE senior_profiles
                                SET status = 'approved',
                                    code_hash = ?
                                WHERE id = ? AND status = 'pending'
                            """, (hash_code(code), sid))

                            st.session_state.new_senior_code = (
                                name, code
                            )
                            st.rerun()

                    with c2:
                        if st.button(
                            "❌ 거절",
                            key=f"senior_reject_{sid}",
                            use_container_width=True
                        ):
                            db_execute("""
                                UPDATE senior_profiles
                                SET status = 'rejected'
                                WHERE id = ? AND status = 'pending'
                            """, (sid,))
                            st.rerun()

            if "new_senior_code" in st.session_state:
                name, code = st.session_state.new_senior_code
                st.success(f"{name} 선배 승인 완료")
                st.caption(
                    "이 코드는 지금만 표시됩니다. "
                    "승인된 선배에게 별도 경로로 전달하세요."
                )
                st.code(code)

                if st.button("코드 확인 완료"):
                    st.session_state.pop("new_senior_code", None)
                    st.rerun()

        with tab2:
            requests = db_query("""
                SELECT r.id, s.name, s.email,
                       r.junior_name, r.junior_email,
                       r.topic, r.message, r.status, r.created_at
                FROM mentoring_requests r
                JOIN senior_profiles s ON r.senior_id = s.id
                ORDER BY r.id DESC
            """)

            if not requests:
                st.info("아직 멘토링 신청이 없습니다.")

            status_options = {
                "pending": "접수",
                "contacted": "연락 완료",
                "completed": "멘토링 완료"
            }

            for (
                rid, mentor_name, mentor_email,
                junior, junior_email, topic,
                message, status, created
            ) in requests:
                with st.expander(
                    f"{junior} → {mentor_name} | {topic}"
                ):
                    st.write(f"멘토: {mentor_name}")
                    st.write(f"멘토 이메일: {mentor_email}")
                    st.write(f"후배: {junior}")
                    st.write(f"후배 이메일: {junior_email}")
                    st.write(f"신청 내용: {message}")
                    st.caption(f"신청일: {created}")

                    values = list(status_options.keys())
                    current = status if status in values else "pending"

                    selected = st.selectbox(
                        "진행 상태",
                        options=values,
                        format_func=lambda v: status_options[v],
                        index=values.index(current),
                        key=f"mentor_status_{rid}"
                    )

                    if st.button(
                        "상태 저장",
                        key=f"mentor_status_save_{rid}"
                    ):
                        db_execute("""
                            UPDATE mentoring_requests
                            SET status = ?
                            WHERE id = ?
                        """, (selected, rid))
                        st.rerun()
