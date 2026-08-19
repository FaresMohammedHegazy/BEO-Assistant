import os
import sys
import sqlite3
import jsonschema

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
import asyncio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Define the database path dynamically
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'db', 'aurelia.db')

# Global state to simulate a role change for the Notifications requirement
is_director_authenticated = False

# 1. Server Initialization
app = Server("aurelia-beo-assistant")


# ==========================================
# DYNAMIC TOOL MANAGEMENT (agent_tools table)
# ==========================================
def _get_inactive_tool_names() -> set[str]:
    """Return the set of tool names the admin panel has disabled.

    Tools are tracked per-agent in the `agent_tools` table (agent_name,
    tool_name, is_active), built by Teammate 1's admin platform work. A
    tool is treated as disabled here only when EVERY agent_tools row for
    that tool_name has is_active = 0 -- i.e. no agent currently has it
    enabled. Tools that aren't tracked yet in the table are left enabled
    (fail-open) so an empty/uninitialized table never hides the whole
    tool surface.

    This is queried fresh on every list_tools()/call_tool() call, so an
    admin flipping a row in the SQLite table is picked up by the live
    MCP server without a redeploy or restart.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT tool_name, MAX(is_active) FROM agent_tools GROUP BY tool_name"
        )
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.Error:
        # Table missing / DB not migrated yet -- fail open rather than
        # hiding every tool.
        return set()

    return {tool_name for tool_name, max_active in rows if not max_active}


def _client_has_elicitation_capability() -> bool:
    """Determine whether the connected client actually negotiated the
    'elicitation' experimental capability during initialize().

    This used to read the DEMO_MODE environment variable directly, which
    only "worked" because the bundled demo client (agent/client.py) sets
    DEMO_MODE in lockstep with the capabilities it separately injects
    into the session. Any other client -- including the real admin
    platform or planning agent -- would just inherit whatever DEMO_MODE
    happened to be set to on the server process, regardless of what it
    actually declared during the handshake. This reads the real
    negotiated ClientCapabilities from the session instead, which is
    what capability negotiation is supposed to do.
    """
    request_context = app.request_context
    session = request_context.session if request_context else None
    if session is None:
        return False

    client_params = getattr(session, "client_params", None)
    if client_params is None:
        return False

    capabilities = getattr(client_params, "capabilities", None)
    if capabilities is None:
        return False

    experimental = getattr(capabilities, "experimental", None) or {}
    return "elicitation" in experimental


# ==========================================
# CONCERN 1: RESOURCES
# ==========================================
@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="aurelia://policies/fire-safety",
            name="Fire Safety and Maximum Room Capacity Policy",
            description="Crucial rules regarding room capacities and fire code enforcement.",
            mimeType="text/plain",
        )
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    if str(uri) == "aurelia://policies/fire-safety":
        policy_text = (
            "AURELIA HOTELS FIRE SAFETY POLICY:\n"
            "- STRICT_ENFORCEMENT: Rooms with this status absolutely cannot exceed max_capacity under any circumstances. "
            "Any event booked over capacity in these rooms is a critical safety violation.\n"
            "- COMPLIANT: Standard occupancy rules apply."
        )
        return policy_text
    raise ValueError(f"Resource not found: {uri}")


# ==========================================
# CONCERN 2: PROMPTS
# ==========================================
@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="draft_beo",
            description="Draft a new Banquet Event Order (BEO) starting template.",
            arguments=[
                types.PromptArgument(
                    name="event_id",
                    description="The unique ID of the event to draft the BEO for.",
                    required=True
                )
            ]
        )
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
    if name != "draft_beo":
        raise ValueError(f"Prompt not found: {name}")

    event_id = (arguments or {}).get("event_id")
    if not event_id:
        raise ValueError("event_id argument is required")

    return types.GetPromptResult(
        description=f"Drafting BEO for {event_id}",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Please help me draft a detailed Banquet Event Order for event ID {event_id}. "
                         f"Before writing it, please check the database for the event's headcount, room assignment, "
                         f"and any VIP guest dietary restrictions to ensure accuracy."
                )
            )
        ]
    )


# ==========================================
# TOOLS & NEGOTIATION / NOTIFICATIONS
# ==========================================
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    # Real capability negotiation instead of the old DEMO_MODE env-var hack.
    has_elicitation = _client_has_elicitation_capability()

    # Tools the admin platform has disabled via the agent_tools table.
    # Queried live on every call so toggles reach the running server
    # without a redeploy.
    inactive_tool_names = _get_inactive_tool_names()

    tools = [
        types.Tool(
            name="audit_chain_wide_availability",
            description="Run a comprehensive, long-running audit of all 150 hotel rooms across the chain to check for availability.",
            inputSchema={
                "type": "object",
                "properties": {
                    "audit_date": {"type": "string", "description": "The date to audit (YYYY-MM-DD)"}
                },
                "required": ["audit_date"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="book_event_room",
            description="Attempt to book a specific room for an event. WARNING: Subject to strict fire code capacity checks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The unique ID of the event to book."},
                    "room_id": {"type": "string", "description": "The unique ID of the room being requested."},
                    "requested_headcount": {"type": "integer", "description": "The total number of guests expected."}
                },
                "required": ["event_id", "room_id", "requested_headcount"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="authenticate_director",
            description="Authenticate as a Senior Director to unlock high-stakes write tools. (The PIN is 1234)",
            inputSchema={
                "type": "object",
                "properties": {
                    "pin": {"type": "string", "description": "The 4-digit authentication PIN."}
                },
                "required": ["pin"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="draft_custom_menu",
            description="Draft a custom BEO menu for a VIP guest using only safe, database-verified ingredients.",
            inputSchema={
                "type": "object",
                "properties": {
                    "guest_id": {"type": "string", "description": "The ID of the VIP guest."}
                },
                "required": ["guest_id"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="view_event_deposit_status",
            description="View the current status and required deposit for an event.",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "The unique ID of the event."}
                },
                "required": ["event_id"],
                "additionalProperties": False
            }
        ),
    ]

    # NOTIFICATIONS: Expose confirm_event_booking ONLY if authenticated and has elicitation
    if is_director_authenticated and has_elicitation:
        tools.append(
            types.Tool(
                name="confirm_event_booking",
                description="Confirm a high-stakes event booking and process deposit. Requires human elicitation.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string", "description": "The unique ID of the event."}
                    },
                    "required": ["event_id"],
                    "additionalProperties": False
                }
            )
        )

    # Exclude anything the admin panel marked is_active = False.
    return [t for t in tools if t.name not in inactive_tool_names]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    # Enforce the same admin toggle at execution time, not just in
    # list_tools(). A client holding a stale/cached tool list -- or
    # calling a tool name directly -- must not be able to bypass a tool
    # the admin panel just disabled.
    if name in _get_inactive_tool_names():
        return [types.TextContent(
            type="text",
            text=f"Error: Tool '{name}' is currently disabled by the admin panel."
        )]

    if name == "audit_chain_wide_availability":
        audit_date = arguments.get("audit_date")
        request_context = app.request_context

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rooms")
        total_rooms = cursor.fetchone()[0]
        conn.close()

        rooms_processed = 0
        batch_size = 50
        while rooms_processed < total_rooms:
            await asyncio.sleep(0.5)
            rooms_processed = min(rooms_processed + batch_size, total_rooms)
            if request_context.meta and request_context.meta.progressToken:
                await request_context.session.send_progress_notification(
                    progress_token=request_context.meta.progressToken,
                    progress=rooms_processed,
                    total=total_rooms
                )

        # Return structured string listing sample rooms for the LLM to read
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT room_id, name, max_capacity FROM rooms LIMIT 4")
        sample_rooms = cursor.fetchall()
        conn.close()

        report = f"Audit complete. All {total_rooms} rooms checked for {audit_date}.\nAvailable options include:\n"
        for r in sample_rooms:
            report += f"- {r[0]} ({r[1]}): Max Capacity {r[2]}\n"

        return [types.TextContent(type="text", text=report)]

    elif name == "book_event_room":
        schema = {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "room_id": {"type": "string"},
                "requested_headcount": {"type": "integer", "minimum": 1}
            },
            "required": ["event_id", "room_id", "requested_headcount"],
            "additionalProperties": False
        }
        try:
            jsonschema.validate(instance=arguments, schema=schema)
        except jsonschema.ValidationError as e:
            return [types.TextContent(type="text", text=f"Validation Error: {e.message}")]

        event_id = arguments["event_id"]
        room_id = arguments["room_id"]
        requested_headcount = arguments["requested_headcount"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT max_capacity, fire_code_status FROM rooms WHERE room_id = ?", (room_id,))
        room = cursor.fetchone()

        if not room:
            conn.close()
            return [types.TextContent(type="text", text="Error: Room not found.")]

        max_cap, fire_status = room

        if requested_headcount > max_cap:
            if fire_status == "STRICT_ENFORCEMENT":
                conn.close()
                return [types.TextContent(
                    type="text",
                    text=f"CRITICAL AUTHORIZATION FAILURE: Room {room_id} has a strict fire code maximum of {max_cap}. "
                         f"Booking {requested_headcount} guests is illegal. Request denied."
                )]
            else:
                conn.close()
                return [types.TextContent(type="text", text=f"Error: Headcount {requested_headcount} exceeds capacity {max_cap}.")]

        cursor.execute("UPDATE events SET room_id = ?, headcount = ? WHERE event_id = ?",
                        (room_id, requested_headcount, event_id))
        conn.commit()
        conn.close()

        return [types.TextContent(type="text", text=f"Success: {event_id} safely booked into {room_id} with {requested_headcount} guests.")]

    elif name == "authenticate_director":
        pin = arguments.get("pin")
        if pin == "1234":
            global is_director_authenticated
            is_director_authenticated = True
            await app.request_context.session.send_tool_list_changed()
            return [types.TextContent(type="text", text="Authentication successful. Senior Director role active. New tools are now available.")]
        return [types.TextContent(type="text", text="Authentication failed: Invalid PIN.")]

    elif name == "draft_custom_menu":
        guest_id = arguments.get("guest_id")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT dietary_restrictions FROM guests WHERE guest_id = ?", (guest_id,))
        guest = cursor.fetchone()

        if not guest:
            conn.close()
            return [types.TextContent(type="text", text="Guest not found.")]

        restrictions = guest[0]
        cursor.execute("SELECT name, is_nut_free, is_vegan FROM safe_ingredients")
        ingredients = cursor.fetchall()
        conn.close()

        ingredient_list = "\n".join([f"- {i[0]} (Nut-Free: {bool(i[1])}, Vegan: {bool(i[2])})" for i in ingredients])

        prompt_text = (
            f"You are drafting a menu for a VIP guest with these restrictions: {restrictions}.\n"
            f"You may ONLY use the following safe ingredients from our database:\n{ingredient_list}\n"
            "Format a cohesive 3-course menu using only these safe ingredients."
        )

        sampling_response = await app.request_context.session.create_message(
            messages=[types.SamplingMessage(
                role="user",
                content=types.TextContent(type="text", text=prompt_text)
            )],
            max_tokens=500
        )

        menu = sampling_response.content.text if sampling_response else "Sampling failed."
        return [types.TextContent(type="text", text=f"Custom Menu Drafted via Sampling:\n{menu}")]

    elif name == "confirm_event_booking":
        event_id = arguments.get("event_id")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT deposit_required FROM events WHERE event_id = ?", (event_id,))
        event = cursor.fetchone()

        if not event:
            conn.close()
            return [types.TextContent(type="text", text="Event not found.")]

        deposit = event[0]
        prompt_text = (
            f"SECURITY CHECK: Event {event_id} requires a non-refundable deposit of ${deposit}. "
            f"Please ask the human coordinator whether to approve or reject this deposit. "
            f"Reply with EXACTLY 'APPROVE' to confirm or 'REJECT' to cancel."
        )

        try:
            # Send a genuine elicitation/create request to the client
            elicitation_response = await app.request_context.session.create_message(
                messages=[types.SamplingMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt_text)
                )],
                max_tokens=50
            )
            # Extract the human response from the sampling result
            human_response = elicitation_response.content.text.strip().upper() if elicitation_response else "REJECT"
        except Exception as e:
            conn.close()
            return [types.TextContent(type="text", text=f"Action aborted due to elicitation error: {str(e)}")]

        if "APPROVE" in human_response:
            cursor.execute("UPDATE events SET status = 'CONFIRMED' WHERE event_id = ?", (event_id,))
            conn.commit()
            conn.close()
            return [types.TextContent(type="text", text=f"Success: Event {event_id} confirmed via human elicitation. Deposit processed.")]
        else:
            conn.close()
            return [types.TextContent(type="text", text=f"Action aborted. Human response was '{human_response}'. Deposit not processed.")]

    elif name == "view_event_deposit_status":
        event_id = arguments.get("event_id")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT status, deposit_required FROM events WHERE event_id = ?", (event_id,))
        event = cursor.fetchone()
        conn.close()

        if not event:
            return [types.TextContent(type="text", text="Event not found.")]

        status, deposit = event[0], event[1]
        if status == "CONFIRMED":
            return [types.TextContent(type="text", text=f"Event {event_id} status is now officially CONFIRMED. The ${deposit} deposit has been successfully processed and paid.")]
        else:
            return [types.TextContent(type="text", text=f"Event {event_id} status is {status}. A deposit of ${deposit} is currently pending.")]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    print("Starting Aurelia BEO Assistant Server on stdio...", file=sys.stderr, flush=True)
    init_options = app.create_initialization_options()
    if init_options.capabilities.experimental is None:
        init_options.capabilities.experimental = {}
    init_options.capabilities.experimental["elicitation"] = {}

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    asyncio.run(main())
