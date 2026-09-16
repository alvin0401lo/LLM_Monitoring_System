def local_explain(result: dict) -> str:
    readings = result["readings"]
    risk = result["failure_probability"]
    factors = []
    if readings["Tool wear [min]"] > 170:
        factors.append("tool wear is elevated")
    if readings["Torque [Nm]"] > 55:
        factors.append("torque is relatively high")
    if readings["Rotational speed [rpm]"] < 1250:
        factors.append("rotational speed is relatively low")
    if not factors:
        factors.append("the current sensor combination matches patterns associated with this risk level")
    factor_text = "; ".join(factors)
    return f"""**Condition**\nThe ML model classifies the machine as **{result['status'].lower()}**, with an estimated failure probability of **{risk:.0%}**. This is a risk estimate, not a confirmed failure.\n\n**Possible factors**\nThe readings suggest that {factor_text}. These signals may indicate increased mechanical load or wear, but they do not establish a root cause.\n\n**Suggested checks**\n1. Inspect the tool and rotating components for wear, resistance, or damage.\n2. Check the mechanical load, torque transmission, and lubrication condition.\n3. Compare the next few readings with the normal operating range before continuing production.\n\n*Deterministic local analysis only. Verify actions against your site's procedures and qualified maintenance staff.*"""


def explain(result: dict, timeout: int = 0) -> tuple[str, bool]:
    """Return local analysis; the LLM integration is intentionally disabled for this prototype."""
    return local_explain(result), False


def answer_local_question(result: dict, question: str) -> str:
    """Answer common demo questions without requiring a language model."""
    question_lower = question.lower()
    impact = result.get("feature_impact", [])
    top_driver = impact[0]["feature"] if impact else "the current sensor combination"
    if "why" in question_lower or "risk" in question_lower or "cause" in question_lower:
        return f"The model estimates {result['failure_probability']:.0%} failure risk. The strongest current model-sensitive signal is **{top_driver}**. This indicates a pattern associated with risk, not a confirmed root cause."
    if "check" in question_lower or "inspect" in question_lower or "action" in question_lower:
        return "Start with the tool condition, torque transmission, and rotating components. Compare the next readings with the normal operating range and follow qualified site procedures."
    if "sensor" in question_lower or "important" in question_lower or "driver" in question_lower:
        return f"The current driver ranking starts with **{top_driver}**. The chart compares each reading with its training-data median and shows model sensitivity, not physical causation."
    return "This prototype uses local rule-based analysis. Ask about the risk, the main sensor driver, or recommended checks for the current reading."
