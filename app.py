import streamlit as st
import hashlib

from common import get_dataset, get_predictor, initialize_state, inject_theme
from llm import ANALYSIS_INTERVAL, ollama_settings, update_analysis
from rag import parse_uploaded_reference

inject_theme()
initialize_state(get_dataset(), get_predictor())
with st.sidebar:
    st.markdown("### NORTHSTAR")
    st.caption("Industrial monitoring / synthetic replay")
    machines = ["M001", "M002", "M003", "M004"]
    st.session_state.machine_id = st.selectbox("Machine", machines,
                                               index=machines.index(st.session_state.machine_id))
    with st.expander("System status"):
        st.markdown(":material/check_circle: ML model ready")
        st.markdown(":material/check_circle: Local rules ready")
        st.caption("Synthetic manuals and maintenance cases available for scoped RAG retrieval.")
    with st.expander("Reference files"):
        files = st.file_uploader("Upload manuals or logs", type=["txt", "md", "csv"],
                                 accept_multiple_files=True, max_upload_size=1, key="reference_uploads")
        choices = {f"{file.name} ({hashlib.sha256(file.getvalue()).hexdigest()[:8]})": file
                   for file in files}
        selected = st.selectbox("Use file for current machine", ["None", *choices],
                                key=f"reference_choice_{st.session_state.machine_id}")
        try:
            st.session_state.selected_uploaded_reference = (
                parse_uploaded_reference(choices[selected].name, choices[selected].getvalue())
                if selected != "None" else None
            )
        except ValueError as error:
            st.session_state.selected_uploaded_reference = None
            st.error(str(error))
        st.caption("Relevant excerpts from the selected file may join the synthetic references. Session only; "
                   "do not upload sensitive records through a public tunnel.")

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:"),
    st.Page("views/live_monitoring.py", title="Live Monitoring", icon=":material/timeline:"),
    st.Page("views/ai_analysis.py", title="AI Analysis", icon=":material/psychology:"),
    st.Page("views/ai_chat.py", title="AI Chat", icon=":material/chat:"),
    st.Page("views/history_alerts.py", title="History & Alerts", icon=":material/history:"),
]

navigation = st.navigation(pages)
navigation.run()


@st.fragment(run_every="5s")
def refresh_ai_analysis() -> None:
    initialize_state(get_dataset(), get_predictor())
    settings = ollama_settings()
    with st.sidebar:
        st.divider()
        st.markdown("**Local LLM analysis**")
        configured = bool(settings["host"] and settings["model"])
        force = st.button("Refresh analysis", icon=":material/refresh:", key="refresh_llm_analysis",
                          disabled=not configured or st.session_state.get("analysis_job") is not None)
        record = update_analysis(st.session_state, st.session_state.result,
                                 st.session_state.history, st.session_state.machine_id, settings, force)
        if not configured:
            st.caption("Ollama not configured")
        elif st.session_state.get("analysis_job"):
            st.caption("Ollama analysis in progress")
        elif record.get("error"):
            st.caption("Ollama unavailable; last successful analysis retained")
        else:
            st.caption(f"Auto analysis every {ANALYSIS_INTERVAL // 60} minutes")
        if record.get("generated_at"):
            st.caption(f"Last analysis: {record['generated_at']}")


refresh_ai_analysis()
