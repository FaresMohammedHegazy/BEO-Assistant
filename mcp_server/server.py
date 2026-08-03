import os
import sys
import sqlite3
import jsonschema
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
import asyncio

# Define the database path dynamically
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'db', 'aurelia.db')

# Global state to simulate a role change for the Notifications requirement
is_director_authenticated = False

# 1. Server Initialization
app = Server("aurelia-beo-assistant")

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