import streamlit as st

from common import get_dataset, get_predictor, initialize_state, inject_theme
from llm import answer_local_question

inject_theme()
dataset = get_dataset()
predictor = get_predictor()
initialize_state(dataset, predictor)
result = st.session_state.result
ANSWER_SOURCES = [
    ("AI4I 2020 dataset", "Current sensor readings and the selected machine row."),
    ("Random Forest model", "Failure probability, risk status, and sensor impact."),
    ("Deterministic local analysis", "Short explanation and general inspection guidance."),
]

st.markdown('<div class="kicker">ENGINEER ASSISTANT / CURRENT MACHINE</div><h1>AI Chat</h1><p class="subtle">Ask focused questions about the latest ML prediction.</p>', unsafe_allow_html=True)

context, chat = st.columns([.8, 1.5])
with context:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Machine context</div>', unsafe_allow_html=True)
        st.markdown(f"**Machine:** {st.session_state.machine_id}")
        st.markdown(f"**Status:** {result['status']}")
        st.markdown(f"**Risk:** {result['failure_probability']:.0%}")
        st.caption("Context is the latest ML result. Open Live Monitoring for sensor detail.")
with chat:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Ask the assistant</div>', unsafe_allow_html=True)
        st.caption("The local assistant explains the ML output. It does not decide failure status.")
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant":
                    with st.expander("Sources used for this answer", expanded=False):
                        for source_name, source_detail in ANSWER_SOURCES:
                            st.markdown(f"- **{source_name}** · {source_detail}")
                        st.caption("Maintenance logs and equipment manuals were not used.")

st.markdown('<div class="section-label">QUICK QUESTIONS</div>', unsafe_allow_html=True)
quick_columns = st.columns(4)
quick_questions = [
    "Why is the risk high?",
    "Which sensor contributes most?",
    "What should I inspect?",
    "Is the condition getting worse?",
]
for column, question in zip(quick_columns, quick_questions):
    if column.button(question, use_container_width=True):
        st.session_state.pending_question = question

question = st.chat_input("Ask about the current machine") or st.session_state.pop("pending_question", None)
if question:
    answer = answer_local_question(result, question)
    st.session_state.chat_messages.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "sources": ANSWER_SOURCES},
    ])
    st.rerun()
