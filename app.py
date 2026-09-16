import streamlit as st

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/dashboard:"),
    st.Page("views/live_monitoring.py", title="Live Monitoring", icon=":material/timeline:"),
    st.Page("views/ai_analysis.py", title="AI Analysis", icon=":material/psychology:"),
    st.Page("views/ai_chat.py", title="AI Chat", icon=":material/chat:"),
    st.Page("views/history_alerts.py", title="History & Alerts", icon=":material/history:"),
]

navigation = st.navigation(pages)
navigation.run()
