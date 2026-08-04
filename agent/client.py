import os
import json
import asyncio
from dotenv import load_dotenv
from groq import AsyncGroq
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import mcp.types as types

load_dotenv()
MODEL = os.getenv("MODEL_NAME")

class BEODemoAgent:
    def __init__(self):
        self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.server_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mcp_server', 'server.py')
        self.messages = []
        self.tools_available = []

    async def chat_with_groq(self, user_prompt, session):
        print(f"\n[User]: {user_prompt}")
        
        system_msg = {
            "role": "system", 
            "content": "You are a helpful BEO assistant. IF a tool returns a menu or specific text, output it EXACTLY as provided. Do not add unverified ingredients."
        }
        
        current_messages = [system_msg] + self.messages
        current_messages.append({"role": "user", "content": user_prompt})
        
        groq_tools = []
        for t in self.tools_available:
            groq_tools.append({
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema}
            })

        groq_tools.append({
            "type": "function",
            "function": {
                "name": "get_prompt_template",
                "description": "Fetch a reusable prompt template from the server.",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "name": {"type": "string"}, 
                        "arguments": {"type": "object"}
                    }, 
                    "required": ["name", "arguments"]
                }
            }
        })
        groq_tools.append({
            "type": "function",
            "function": {
                "name": "read_resource",
                "description": "Fetch a static resource from the server.",
                "parameters": {"type": "object", "properties": {"uri": {"type": "string"}}, "required": ["uri"]}
            }
        })
        groq_tools.append({
            "type": "function",
            "function": {
                "name": "get_available_tools",
                "description": "Get a list of all currently available tools.",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        })

        response = await self.groq_client.chat.completions.create(
            model=MODEL,
            messages=current_messages,
            tools=groq_tools,
            temperature=0.2
        )
        
        response_message = response.choices[0].message
        
        if response_message.tool_calls:
            self.messages.append({"role": "user", "content": user_prompt})
            self.messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                print(f"   [LLM calling tool]: {func_name}({args})")
                
                try:
                    if func_name == "get_prompt_template":
                        prompt_args = args.get("arguments", {})
                        if not prompt_args and "event_id" in args:
                            prompt_args = {"event_id": args["event_id"]}
                        result = await session.get_prompt(name=args.get("name"), arguments=prompt_args)
                        result_text = result.messages[0].content.text
                    elif func_name == "read_resource":
                        result = await session.read_resource(uri=args["uri"])
                        result_text = result if isinstance(result, str) else result.contents[0].text
                    elif func_name == "get_available_tools":
                        result_text = "Current Tools:\n" + "\n".join([f"- {t.name}: {t.description}" for t in self.tools_available])
                    else:
                        result = await session.call_tool(func_name, arguments=args)
                        result_text = result.content[0].text
                    
                    print(f"   [Tool Result]: {result_text}")
                    self.messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": func_name, "content": result_text})
                except Exception as e:
                    print(f"   [Tool Error]: {str(e)}")
                    self.messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": func_name, "content": f"Error: {str(e)}"})
            
            final_response = await self.groq_client.chat.completions.create(
                model=MODEL,
                messages=[system_msg] + self.messages,
                tools=groq_tools,
                temperature=0.2
            )
            final_text = final_response.choices[0].message.content
            print(f"\n[Agent]: {final_text}")
            self.messages.append({"role": "assistant", "content": final_text})
        else:
            self.messages.append({"role": "user", "content": user_prompt})
            final_text = response_message.content
            print(f"\n[Agent]: {final_text}")
            self.messages.append({"role": "assistant", "content": final_text})

    async def run_fallback_demo(self):
        print("\n" + "="*65)
        print("STEP 1: THE FALLBACK DEMO (Connecting without Elicitation)")
        print("="*65)
        
        env = os.environ.copy()
        env["DEMO_MODE"] = "fallback"
        server_params = StdioServerParameters(command="python", args=[self.server_path], env=env)
        
        async with stdio_client(server_params) as (read_stream, write_stream):
            # Client explicitly initialized WITHOUT capabilities
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print("[Client System] Connected to server WITHOUT 'elicitation' capability.")
                
                print("[Client System] Authenticating to trigger tool load...")
                await session.call_tool("authenticate_director", arguments={"pin": "1234"})
                
                tools_result = await session.list_tools()
                tool_names = [t.name for t in tools_result.tools]
                print(f"[Client System] Available Tools: {tool_names}")
                
                if "view_event_deposit_status" in tool_names and "confirm_event_booking" not in tool_names:
                    print("[SUCCESS] Server safely fell back to read-only tool because we lack elicitation!")
                else:
                    print("[FAILED] Fallback logic didn't work as expected.")
                
        print("[Client System] Disconnected. Server state reset.\n")

    async def run_main_demo(self):
        print("\n" + "="*65)
        print("STEP 2-8: THE MAIN DEMO (Full Handshake & Execution)")
        print("="*65)
        
        # Force the server into main demo mode
        env = os.environ.copy()
        env["DEMO_MODE"] = "main"
        server_params = StdioServerParameters(command="python", args=[self.server_path], env=env)
        
        async with stdio_client(server_params) as (read_stream, write_stream):
            
            async def handle_sampling(context, params) -> types.CreateMessageResult:
                prompt = params.messages[0].content.text
                
                # Intercept the security check for human elicitation
                is_elicitation_trigger = "SECURITY CHECK" in prompt
                if is_elicitation_trigger:
                    print("\n   [Elicitation Alert]: Server paused execution for human approval!")
                    print(f"   [Server Prompt]: {prompt}")
                    human_elicitation_response = input("   [Your Input (Type 'APPROVE' to confirm or 'REJECT' to cancel)]: ").strip().upper()
                    return types.CreateMessageResult(
                        role="assistant",
                        content=types.TextContent(type="text", text=human_elicitation_response),
                        model=MODEL
                    )
                
                # Standard sampling for menu generation
                print("\n   [Sampling Alert]: Server requested the LLM to reason over raw DB facts!")
                print(f"   [Server Prompt]:\n{prompt}")
                
                response = await self.groq_client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=500
                )
                return types.CreateMessageResult(
                    role="assistant",
                    content=types.TextContent(type="text", text=response.choices[0].message.content),
                    model=MODEL
                )

            async with ClientSession(
                read_stream, 
                write_stream,
                sampling_callback=handle_sampling
            ) as session:
                
                # Inject client capability for elicitation
                if hasattr(session, "_client_capabilities"):
                    session._client_capabilities = types.ClientCapabilities(
                        experimental={"elicitation": {}, "sampling": {}}
                    )

                init_result = await session.initialize()
                
                server_caps = init_result.capabilities
                if server_caps.experimental and "elicitation" in server_caps.experimental:
                    print("\n[Capability Negotiation]: Client verified Server supports 'elicitation'. Safe to proceed.")
                
                tools_result = await session.list_tools()
                self.tools_available = tools_result.tools

                demo_prompts = [
                    "Fetch the 'draft_beo' prompt template for event_id 'EVT_999'.",
                    "Fetch the resource at 'aurelia://policies/fire-safety' to check the rules for STRICT_ENFORCEMENT.",
                    "Attempt to book EVT_999 into ROOM_101 with 500 guests. (This should fail due to fire code).",
                    "Run the chain-wide availability audit for 2026-10-31 to find a different room. (Watch for progress delays)",
                    "Draft a custom menu for GUEST_VIP_1.",
                    
                    # --- BEFORE CHECK ---
                    "Before we authenticate, check the current deposit status for EVT_999.", 
                    
                    "Authenticate as Senior Director using PIN 1234.",
                    "Now that we are authenticated, please explicitly fetch the tool list using get_available_tools.",
                    "Confirm the event booking for EVT_999 to process the deposit.",
                    
                    # --- AFTER CHECK ---
                    "Check the deposit status for EVT_999 one more time to verify the database changed." 
                ]
                
                for prompt in demo_prompts:
                    await self.chat_with_groq(prompt, session)
                    await asyncio.sleep(1.0)
                    
                    if "Authenticate" in prompt or "authenticated" in prompt:
                         updated_tools = await session.list_tools()
                         self.tools_available = updated_tools.tools

async def main():
    agent = BEODemoAgent()
    await agent.run_fallback_demo()
    await agent.run_main_demo()

if __name__ == "__main__":
    asyncio.run(main())