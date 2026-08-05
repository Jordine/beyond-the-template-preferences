"""Deterministic segmentation shared by the user-sim judge + analysis.

Close/turn-marker detection and garbled checks are string facts, not
judgment calls — computed once here so judge and analysis cannot drift.
"""
import re

FAM_CLOSE = {"llama": "<|eot_id|>", "qwen": "<|im_end|>", "gemma": "<end_of_turn>"}
FAM_AOPEN = {"llama": "<|start_header_id|>assistant",
             "qwen": "<|im_start|>assistant", "gemma": "<start_of_turn>model"}
ROLE_LINE = re.compile(r"(?mi)^[ \t]*(assistant|ai|llm|chatgpt|human|user)[ \t]*:")
ROLE_ASSIST = re.compile(r"(?mi)^[ \t]*(assistant|ai|llm|chatgpt)[ \t]*:")
CLOSE_NEAR = 40


def family_of(short):
    for fam in ("llama", "qwen", "gemma"):
        if short.startswith(fam):
            return fam
    raise ValueError(short)


def first_close(short, text):
    """(char_pos, kind) of the first turn-close event, or (None, None).
    Any special close/open token or role line ends the current user turn."""
    fam = family_of(short)
    events = []
    for kind, tok in (("fam_close", FAM_CLOSE[fam]), ("fam_aopen", FAM_AOPEN[fam])):
        i = text.find(tok)
        if i != -1:
            events.append((i, kind))
    m = ROLE_LINE.search(text)
    if m:
        events.append((m.start(), "role_line"))
    return min(events) if events else (None, None)


def is_garbled(text):
    words = text.split()
    if len(words) >= 30 and len(set(words)) / len(words) < 0.2:
        return True
    s = text.strip()
    return len(s) >= 40 and len(set(s)) <= 3


def segment(short, text):
    """Cut the continuation at the first turn marker.

    Returns dict:
      close_pos / close_kind — first marker (None if none in 200 tokens)
      pre_seg     — text before the marker (or all of it)
      pre_garbled — deterministic garble check on pre_seg (full text if no close)
      needs_judge — True iff register must be judged by the LLM:
                    not garbled, and no near-immediate close (close > 40 chars
                    or absent), and a non-trivial fragment exists.
      mech_register — composed label when needs_judge is False, else None
                    (GARBLED, or CLOSE_THEN_ASSISTANT for close <= 40 chars).
    """
    pos, kind = first_close(short, text)
    pre = text if pos is None else text[:pos]
    garbled = is_garbled(pre)
    out = {"close_pos": pos, "close_kind": kind, "pre_seg": pre,
           "pre_garbled": garbled}
    if garbled:
        out.update(needs_judge=False, mech_register="GARBLED")
    elif pos is not None and pos <= CLOSE_NEAR:
        out.update(needs_judge=False, mech_register="CLOSE_THEN_ASSISTANT")
    elif len(pre.strip()) < 3:
        out.update(needs_judge=False, mech_register="GARBLED")
    else:
        out.update(needs_judge=True, mech_register=None)
    return out


JUDGE_TO_FINAL = {"USER_REGISTER": "USER_VOICE",
                  "ASSISTANT_REGISTER": "ASSISTANT_VOICE",
                  "MIXED_OR_UNCLEAR": "MIXED", "GARBLED": "GARBLED"}


def compose_register(seg, judge_register):
    """Final 5-way register from deterministic segment + judge fragment label."""
    if not seg["needs_judge"]:
        return seg["mech_register"]
    if judge_register is None:
        return None
    return JUDGE_TO_FINAL.get(judge_register)
