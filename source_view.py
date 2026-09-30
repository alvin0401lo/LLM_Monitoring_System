"""Show the exact reference excerpts supplied to an analysis or chat answer."""

import streamlit as st


def show_sources(documents: list[dict], report: dict | None = None) -> None:
    if report and report.get("invalid"):
        st.warning("Unverified source identifiers: " + ", ".join(report["invalid"]))
    if report and report.get("uncited"):
        st.warning("The answer has no valid reference citation. Treat reference-based claims as unverified.")
    st.caption("Built-in references are synthetic; uploaded references are unverified. Retrieval is not proof of a diagnosis.")
    if not documents:
        st.info("No relevant reference excerpt was found for this machine and query.")
    for document in documents:
        cited = report and document["source_id"] in report.get("cited", [])
        with st.expander(f"{document['source_id']} | {document['file']}" + (" | Cited" if cited else "")):
            st.caption(f"{'SYNTHETIC' if document['synthetic'] else 'UPLOADED'} | "
                       f"{document['machine_id']} | {document['equipment_model']} | "
                       f"{document.get('date', 'demo-1')} | Keyword score {document['score']:.3f}")
            st.text(document["text"])
