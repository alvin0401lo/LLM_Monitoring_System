import streamlit as st

from common import get_dataset, get_predictor, initialize_state, inject_theme
from llm import OllamaError, ask_question, machine_context, ollama_settings
from rag import citation_report, prepare_context
from source_view import show_sources

inject_theme()
dataset = get_dataset()
predictor = get_predictor()
initialize_state(dataset, predictor)
result = st.session_state.result
settings = ollama_settings()
configured = bool(settings["host"] and settings["model"])

st.markdown('<div class="kicker">ENGINEER ASSISTANT / CURRENT MACHINE</div><h1>AI Chat</h1><p class="subtle">Ask focused questions about the latest ML prediction.</p>', unsafe_allow_html=True)

metrics = st.columns(3)
metrics[0].metric("Machine", st.session_state.machine_id)
metrics[1].metric("Failure probability", f"{result['failure_probability']:.0%}")
metrics[2].metric("Risk status", result["status"])
if configured:
    st.caption(f"Ollama / {settings['model']} | Latest machine snapshot")
else:
    st.info("Ollama is not configured.")

st.markdown('<div class="section-label">QUICK QUESTIONS</div>', unsafe_allow_html=True)
quick_questions = [
    "Why is the risk high?",
    "Which sensor contributes most?",
    "What should I inspect?",
    "Is the condition getting worse?",
]
with st.container(horizontal=True):
    for question in quick_questions:
        if st.button(question, icon=":material/chat_bubble:", disabled=not configured):
            st.session_state.pending_question = question

st.divider()
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
machine_messages = [message for message in st.session_state.chat_messages
                    if message.get("machine_id") == st.session_state.machine_id]
if not machine_messages:
    st.markdown("### Ask the assistant")
    st.caption("Decision support only. Final operational actions require qualified review.")
for message in machine_messages:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.write(message["content"])
        if message["role"] == "assistant" and not message.get("error"):
            st.caption(f"Ollama / {message['model']} | Data snapshot: {message['data_time']}")
            with st.expander("Evidence supplied for this answer", expanded=False):
                for source_name, source_detail in message.get("sources", []):
                    st.markdown(f"- **{source_name}** · {source_detail}")
                show_sources(message.get("retrieved_documents", []), message.get("citations"))

question = st.chat_input("Ask about the current machine", disabled=not configured) or st.session_state.pop("pending_question", None)
if question and configured:
    machine_id = st.session_state.machine_id
    snapshot = prepare_context(machine_context(st.session_state.result, st.session_state.history, machine_id), question)
    conversation = [message for message in st.session_state.chat_messages if message.get("machine_id") == machine_id]
    sources = [
        ("AI4I synthetic replay", "Sensor snapshot and available recent replay summary."),
        ("Random Forest model", "Estimated failure probability and feature sensitivity."),
    ]
    error = False
    try:
        with st.spinner("Waiting for Ollama..."):
            answer = ask_question(snapshot, question, conversation, settings)
    except OllamaError as failure:
        answer = str(failure)
        error = True
    st.session_state.chat_messages.extend([
        {"role": "user", "content": question, "machine_id": machine_id},
        {"role": "assistant", "content": answer, "sources": sources, "machine_id": machine_id,
         "model": settings["model"], "data_time": snapshot["data_time"], "error": error,
         "retrieved_documents": snapshot["retrieved_documents"],
         "citations": citation_report(answer, snapshot["retrieved_documents"])},
    ])
    st.rerun()
