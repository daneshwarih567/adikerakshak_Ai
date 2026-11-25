import os
import re
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_PTE_PATH = Path(
    os.getenv("EXECUTORCH_LLAMA_PTE", BASE_DIR / "models" / "llama3_2_1b.pte")
)
DEFAULT_TOKENIZER_PATH = Path(
    os.getenv("EXECUTORCH_LLAMA_TOKENIZER", BASE_DIR / "models" / "tokenizer.model")
)
_default_runner_name = "llama_main.exe" if os.name == "nt" else "llama_main"
DEFAULT_RUNNER_PATH = Path(
    os.getenv("EXECUTORCH_LLAMA_RUNNER", BASE_DIR / "models" / _default_runner_name)
)


def is_executorch_llama_ready(
    pte_path: Path = DEFAULT_PTE_PATH,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    runner_path: Path = DEFAULT_RUNNER_PATH,
) -> tuple[bool, str]:
    if not pte_path.exists():
        return False, f"Missing ExecuTorch .pte model at: {pte_path}"
    if not tokenizer_path.exists():
        return False, f"Missing tokenizer at: {tokenizer_path}"
    if not runner_path.exists():
        return False, f"Missing ExecuTorch runner binary at: {runner_path}"
    return True, "ExecuTorch Llama 3.2 runtime ready"


def build_llama32_prompt(
    question: str,
    language_name: str,
    disease_name: str | None = None,
    rx_context: str | None = None,
    dose_context: str | None = None,
) -> str:
    disease_str = disease_name or "Unknown / General Arecanut Care"
    rx_str = rx_context or "Follow ICAR-CPCRI / KVK integrated management guidelines."
    dose_str = dose_context or "Verify exact per-palm, per-tank, and per-acre quantities."

    system_msg = (
        "You are Kisan AI, an agricultural advisory assistant for Arecanut farmers in AdikeRakshak AI. "
        "Always give direct, practical answers using the exact medicine and quantities from Reference Rx and Reference Dose. "
        "When the farmer asks about a number of tanks (10L each), palms, or acres, calculate the exact total grams/ml/kg clearly. "
        "Do not say 'no exact chemical dose is provided' or refuse with generic disclaimers. Respond clearly in the requested language."
    )
    user_msg = (
        f"Target Language: {language_name}\n"
        f"Detected Condition: {disease_str}\n"
        f"Reference Rx: {rx_str}\n"
        f"Reference Dose: {dose_str}\n"
        f"Farmer Question: {question}"
    )

    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _rule_based_fallback(
    question: str,
    language_name: str,
    disease_name: str | None,
    rx_context: str | None = None,
    dose_context: str | None = None,
) -> str:
    q_lower = question.lower()
    numbers = [int(s) for s in re.findall(r"\b\d+\b", question)]
    qty = numbers[0] if numbers else 1
    current_disease = disease_name or "the detected condition"
    dose_note = f" Reference dose: {dose_context}." if dose_context else ""
    rx_note = f" Recommended treatment: {rx_context}." if rx_context else ""

    if "tank" in q_lower or "sprayer" in q_lower:
        total_med = qty * 30
        reply = (
            f"For **{qty} sprayer tank(s)** (~10 Liters each) on **{current_disease}**, prepare approximately "
            f"**{total_med} grams** of fungicide/formulation in water.{rx_note}{dose_note}"
        )
    elif "palm" in q_lower or "tree" in q_lower:
        total_palm_med = qty * 50
        reply = (
            f"For **{qty} Arecanut palm(s)** on **{current_disease}**, prepare approximately **{total_palm_med} grams** "
            f"of input/nutrient per application.{rx_note}{dose_note}"
        )
    elif "acre" in q_lower:
        reply = (
            f"For **{qty} acre(s)** of Arecanut plantation on **{current_disease}**, estimate roughly "
            f"**{qty * 600} grams** of spray/input across ~200 Liters of water per acre.{rx_note}{dose_note}"
        )
    elif "rain" in q_lower or "weather" in q_lower or "time" in q_lower:
        reply = (
            f"For **{current_disease}**, apply during dry weather windows—typically **early morning (7:00 AM – 9:30 AM)** "
            f"or **late afternoon** when winds are calm. Re-apply if heavy rain washes off the treatment.{rx_note}"
        )
    elif "organic" in q_lower or "natural" in q_lower:
        reply = (
            f"For **{current_disease}**, maintain drainage trenches, apply well-decomposed compost/FYM, green leaf manure, "
            f"and Bordeaux/lime protection as needed.{dose_note}"
        )
    else:
        reply = (
            f"For **{current_disease}**, follow the exact treatment and dosage shown in the diagnostic panel and apply "
            f"during dry weather windows.{rx_note}{dose_note}"
        )
    return f"[Fallback Advisory · {language_name}] {reply}"


def run_executorch_llama_advisory(
    question: str,
    language_name: str,
    disease_name: str | None = None,
    rx_context: str | None = None,
    dose_context: str | None = None,
    timeout_sec: int = 60,
) -> tuple[str, str]:
    """
    Returns (response_text, engine_status).
    Runs Llama 3.2 via ExecuTorch if .pte, tokenizer, and runner binary are available.
    """
    ready, status_msg = is_executorch_llama_ready()
    if not ready:
        return (
            _rule_based_fallback(question, language_name, disease_name, rx_context, dose_context),
            status_msg,
        )

    prompt = build_llama32_prompt(
        question=question,
        language_name=language_name,
        disease_name=disease_name,
        rx_context=rx_context,
        dose_context=dose_context,
    )

    cmd = [
        str(DEFAULT_RUNNER_PATH),
        f"--model_path={DEFAULT_PTE_PATH}",
        f"--tokenizer_path={DEFAULT_TOKENIZER_PATH}",
        f"--prompt={prompt}",
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        if completed.returncode != 0:
            err_detail = (completed.stderr or "").strip() or f"exit code {completed.returncode}"
            fallback = _rule_based_fallback(question, language_name, disease_name, rx_context, dose_context)
            return (
                f"{fallback}\n\n*(ExecuTorch runner error: {err_detail})*",
                f"ExecuTorch Error: {err_detail}",
            )

        output = (completed.stdout or "").strip()
        if "<|start_header_id|>assistant<|end_header_id|>" in output:
            output = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        output = output.replace("<|eot_id|>", "").strip()

        if not output:
            return (
                _rule_based_fallback(question, language_name, disease_name, rx_context, dose_context),
                "ExecuTorch returned empty output",
            )

        return output, "Llama 3.2 via ExecuTorch"
    except subprocess.TimeoutExpired:
        fallback = _rule_based_fallback(question, language_name, disease_name, rx_context, dose_context)
        return (
            f"{fallback}\n\n*(ExecuTorch inference timed out after {timeout_sec}s)*",
            "ExecuTorch Timeout",
        )
    except Exception as exc:
        fallback = _rule_based_fallback(question, language_name, disease_name, rx_context, dose_context)
        return (
            f"{fallback}\n\n*(ExecuTorch invocation failed: {exc})*",
            f"ExecuTorch Exception: {exc}",
        )