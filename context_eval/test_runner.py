import json
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "test_suite.json")
LENGTHS = [20, 40, 60]

CRITICAL_FACTS = [
    {
        "name": "nut_allergy",
        "statement": "Just a note for the file: Eleanor Vance (GUEST_VIP_1) has a SEVERE NUT ALLERGY and is VEGAN. Please always flag this for her events.",
        "ack": "Noted. I've flagged Eleanor Vance's severe nut allergy and vegan requirement for all future menu planning.",
        "final_question": "We're finalizing the menu for EVT_999 now. Any dietary restrictions to watch for?",
        "expected_fact_snippet": "nut allerg"
    },
    {
        "name": "fire_code_capacity",
        "statement": "Important note: ROOM_101 (Grand Magnolia Ballroom) has a STRICT_ENFORCEMENT fire code cap of exactly 300 guests. This can never be exceeded under any circumstances.",
        "ack": "Understood. ROOM_101's 300-guest fire code maximum is strictly enforced and noted.",
        "final_question": "We'd like to raise EVT_999's headcount to 350 in ROOM_101 — is that possible?",
        "expected_fact_snippet": "300"
    },
    {
        "name": "deposit_requirement",
        "statement": "For the record: EVT_999 requires a non-refundable deposit of $20,000 before the booking can be confirmed.",
        "ack": "Noted. EVT_999's $20,000 non-refundable deposit requirement is on file.",
        "final_question": "What are the financial conditions to finalize confirmation of EVT_999?",
        "expected_fact_snippet": "20,000"
    }
]


def generate_noisy_tool_result(index):
    rooms = "\n".join([f"- ROOM_{100+i}: Standard Conference Room {i}, Capacity 50" for i in range(2, 40)])
    return {"role": "tool", "content": f"Audit batch {index} complete.\n{rooms}",
            "tool_call_id": f"call_{index}", "name": "audit_chain_wide_availability"}


def generate_filler_dialogue(index):
    return [
        {"role": "user", "content": f"Can you check room availability for a small meeting #{index}?"},
        {"role": "assistant", "content": f"Sure, checking availability now for request #{index}."}
    ]


def build_test_case(fact, total_turns):
    messages = [
        {"role": "system", "content": "You are a helpful BEO assistant."},
        {"role": "user", "content": fact["statement"]},
        {"role": "assistant", "content": fact["ack"]},
    ]
    turn = len(messages)
    while turn < total_turns - 1:
        messages.extend(generate_filler_dialogue(turn))
        messages.append(generate_noisy_tool_result(turn))
        turn = len(messages)
    messages.append({"role": "user", "content": fact["final_question"]})

    return {
        "name": f"{fact['name']}_buried_{total_turns}turns",
        "critical_fact_turn": 1,
        "final_question": fact["final_question"],
        "expected_fact_snippet": fact["expected_fact_snippet"],
        "messages": messages
    }

def run():
    cases = [build_test_case(fact, length) for fact in CRITICAL_FACTS for length in LENGTHS]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(cases)} test cases → {OUTPUT_PATH}")
    return cases

if __name__ == "__main__":
    run()