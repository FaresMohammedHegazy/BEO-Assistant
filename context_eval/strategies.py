import os

try:
    from groq import AsyncGroq
except Exception:  # pragma: no cover - runtime dependency is optional for offline metrics
    AsyncGroq = None

# --- HELPER FUNCTIONS ---
# The agent buffer stores both dictionaries and Groq ChoiceMessage objects.
# These helpers ensure we can read and clone messages safely regardless of their type.

def _get_role(msg):
    return msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)

def _get_content(msg):
    return msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)

def _clone_and_mask(msg, new_content):
    if isinstance(msg, dict):
        new_msg = msg.copy()
    else:
        new_msg = {"role": getattr(msg, "role", ""), "content": getattr(msg, "content", "")}
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            new_msg["tool_calls"] = msg.tool_calls
            
    new_msg["content"] = new_content
    return new_msg


# --- STRATEGY 1: SLIDING WINDOW ---
def apply_sliding_window(messages, window_size=10):
    """
    Keeps the system persona and strictly drops older dialogue 
    outside of the immediate rolling N-turn window.
    """
    if not messages:
        return []
    
    system_msgs = [m for m in messages if _get_role(m) == "system"]
    history_msgs = [m for m in messages if _get_role(m) != "system"]
    
    return system_msgs + history_msgs[-window_size:]


# --- STRATEGY 2: OBSERVATION AND TOOL-OUTPUT MASKING ---
def apply_observation_masking(messages, max_unmasked_tools=3):
    """
    Preserves all dialogue turns but heavily truncates large 
    JSON tool outputs beyond the last K tool calls.
    """
    pruned_messages = []
    tool_count = 0
    
    for msg in reversed(messages):
        if _get_role(msg) == "tool":
            tool_count += 1
            if tool_count > max_unmasked_tools:
                # Mask the payload to save tokens while keeping the tool context alive
                masked_msg = _clone_and_mask(msg, "[TOOL OUTPUT MASKED - PREVIOUSLY PROCESSED]")
                pruned_messages.insert(0, masked_msg)
            else:
                pruned_messages.insert(0, msg)
        else:
            pruned_messages.insert(0, msg)
            
    return pruned_messages


# --- STRATEGY 3: RECURSIVE SUMMARIZATION ---
async def apply_recursive_summarization(messages, compact_every=15, client=None, model=None):
    """
    Uses an LLM pass to compress older dialogue into a tight summary 
    once the context window hits a specific size threshold.
    """
    if not messages:
        return []

    system_msgs = [m for m in messages if _get_role(m) == "system"]
    history_msgs = [m for m in messages if _get_role(m) != "system"]

    if len(history_msgs) <= compact_every:
        return messages

    older_msgs = history_msgs[:-10]
    recent_msgs = history_msgs[-10:]

    dialogue_to_summarize = ""
    for m in older_msgs:
        role = _get_role(m) or "unknown"
        content = _get_content(m) or "tool_call_issued"
        dialogue_to_summarize += f"{role.upper()}: {content}\n"

    prompt = (
        "Summarize the following interaction concisely. "
        "Preserve all specific facts, extracted database identifiers (like EVT_999 or ROOM_101), "
        "and active tasks.\n\n"
        f"{dialogue_to_summarize}"
    )

    if not client:
        api_key = os.getenv("GROQ_API_KEY")
        if AsyncGroq is None or not api_key:
            summary_text = " ".join(
                f"{_get_role(m) or 'unknown'}: {_get_content(m) or 'tool_call_issued'}"
                for m in older_msgs
            )
            summary_msg = {
                "role": "system",
                "content": f"PRIOR CONTEXT SUMMARY: {summary_text[:1400]}",
            }
            return system_msgs + [summary_msg] + recent_msgs

        client = AsyncGroq(api_key=api_key)

    if not model:
        model = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        summary = response.choices[0].message.content
    except Exception:
        summary = dialogue_to_summarize[:1400]

    summary_msg = {"role": "system", "content": f"PRIOR CONTEXT SUMMARY: {summary}"}

    return system_msgs + [summary_msg] + recent_msgs


# --- STRATEGY 4: ZONE-BASED PRUNING ---
def apply_zone_based_pruning(messages, keep_first_n=2, keep_last_n=6):
    """
    Divides context into 4 zones, actively dropping the 'middle' dialogue 
    (Zone 3) while preserving the critical early facts (Zone 2).
    """
    if not messages:
        return []
        
    system_msgs = [m for m in messages if _get_role(m) == "system"]
    history_msgs = [m for m in messages if _get_role(m) != "system"]
    
    if len(history_msgs) <= (keep_first_n + keep_last_n):
        return messages
        
    zone_2_early_facts = history_msgs[:keep_first_n]
    zone_4_recent_context = history_msgs[-keep_last_n:]
    
    dropped_count = len(history_msgs) - keep_first_n - keep_last_n
    zone_3_placeholder = {
        "role": "system", 
        "content": f"[ZONE PRUNING APPLIED: {dropped_count} intermediate messages removed to save context]"
    }
    
    return system_msgs + zone_2_early_facts + [zone_3_placeholder] + zone_4_recent_context