import os
import json
import asyncio
import sys
from dotenv import load_dotenv
from groq import AsyncGroq
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import mcp.types as types
from memory.short_term import ShortTermMemory, Message
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from memory.router import MemoryRouter
from memory.consolidation import SemanticConsolidator

from rag.vector_store import VectorStore
from rag.retrievers import HybridRAG
from rag.self_rag import SelfRAG
load_dotenv()
MODEL = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))

class BEODemoAgent:
    def __init__(self):
        api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. export GROQ_API_KEY=<your Groq key> before running agent/client.py."
            )
        if "\n" in api_key or "\r" in api_key or api_key.startswith("export ") or api_key.startswith("cd "):
            raise RuntimeError(
                "GROQ_API_KEY looks malformed. It appears to contain shell text instead of a real Groq API key. "
                "Unset the bad environment variable and export GROQ_API_KEY=<your Groq key> only."
            )
        self.groq_client = AsyncGroq(api_key=api_key)
        self.server_path = os.path.join(REPO_ROOT, 'mcp_server', 'server.py')

        # --- Long-term memory stack ---
        self.episodic_store = EpisodicStore()
        self.semantic_store = SemanticStore()
        self.memory_router = MemoryRouter(episodic_store=self.episodic_store)
        self.consolidator = SemanticConsolidator(self.semantic_store, self.episodic_store)

        self.stm = ShortTermMemory(buffer_size=8, router=self.memory_router)
        self._turns_since_consolidation = 0
        self._consolidate_every_n_turns = 3

        # --- RAG stack ---
        rag_db_path = os.path.join(REPO_ROOT, 'db', 'rag_store.sqlite')
        self.vector_store = VectorStore(store_path=rag_db_path)
        self.vector_store.initialize()
        self.hybrid_rag = HybridRAG(self.vector_store)
        self.self_rag = SelfRAG()

        self._critical_keywords = ("allergy", "allergic", "fire code", "deposit", "nut")
   
        self.tools_available = []

    def _to_groq_format(self, msg: Message) -> dict:
        # The Groq chat API accepts a narrowed OpenAI-ish schema:
        #   - role/content for normal messages
        #   - tool_call_id for tool-role result messages
        #   - tool_calls for assistant messages that launched tool calls
        # Memory-only metadata such as {important, persist, event_id, task_id,
        # name, source, etc.} is kept off the outbound wire contract.
        payload = {
            "role": msg.role,
            "content": msg.content or "",
        }

        if msg.role == "tool":
            tool_call_id = msg.metadata.get("tool_call_id") if msg.metadata else None
            if tool_call_id:
                payload["tool_call_id"] = tool_call_id
        elif msg.role == "assistant" and msg.metadata:
            tool_calls = msg.metadata.get("tool_calls")
            if tool_calls:
                payload["tool_calls"] = tool_calls

        return payload

    def _metadata_for_user_message(self, content: str) -> dict:
        lowered = (content or "").lower()
        if any(keyword in lowered for keyword in self._critical_keywords):
            return {"important": True}
        return {}

    async def _maybe_run_consolidation(self):
        self._turns_since_consolidation += 1
        if self._turns_since_consolidation < self._consolidate_every_n_turns:
            return
        self._turns_since_consolidation = 0
        print("\n   [Memory] Running periodic consolidation pass over episodic memory...")
        await self.consolidator.run_consolidation_pass()
    async def _run_rag_search(self, query: str) -> str:
        docs = self.hybrid_rag.retrieve(query, top_k=3)
        is_relevant, rel_reason = self.self_rag.evaluate_retrieval(query, docs)
        if not is_relevant:
            return "No sufficiently relevant policy information was found for that question."

        context = "\n".join(f"- {d['text']}" for d in docs)
        draft_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer using ONLY the context above."
        draft_response = await self.groq_client.chat.completions.create(
             model=MODEL, messages=[{"role": "user", "content": draft_prompt}], temperature=0.0,
        )
        draft_answer = draft_response.choices[0].message.content

        is_supported, sup_reason = self.self_rag.evaluate_support(query, draft_answer, docs)
        if not is_supported:
            return ("I found related policy text, but couldn't verify the drafted answer was fully "
                "supported by it. Retrieved context: " + context)
        return draft_answer

    async def _run_memory_recall(self, guest_id: str) -> str:
        facts = self.semantic_store.get_active_facts(guest_id)
        if not facts:
            return f"No durable facts on file yet for {guest_id}."

        pseudo_docs = [{"text": f"{attr}: {val}"} for attr, val in facts.items()]
        is_relevant, _ = self.self_rag.evaluate_retrieval(guest_id, pseudo_docs)
        if not is_relevant:
            return f"Recalled facts for {guest_id} did not pass relevance verification; withholding them."

        summary = ", ".join(f"{attr}={val}" for attr, val in facts.items())
        is_supported, _ = self.self_rag.evaluate_support(f"What do we know about {guest_id}?", summary, pseudo_docs)
        if not is_supported:
            return f"Recalled facts for {guest_id} failed the support check; withholding them."

        return f"Semantic memory on {guest_id}: {summary}"

    def _recover_empty_final_text(self, user_prompt: str, tool_results: list[str]) -> str:
        """Return a user-visible explanation when the Groq final completion
        comes back empty or a tool-only turn is unanswerable from evidence.
        """
        evidence = [r for r in tool_results if r and str(r).strip()]
        if evidence:
            snippet = "\n".join(evidence[:3])
            return (
                f"I could not synthesize a grounded answer from the current evidence. "
                f"The available search results were:\n{snippet}"
            )
        return (
            "I could not synthesize a grounded answer from the current evidence. "
            "The retrieval phase returned no supporting policy content."
        )

    async def chat_with_groq(self, user_prompt, session):
        print(f"\n[User]: {user_prompt}")
        
        system_msg = {
            "role": "system", 
            "content": "You are a helpful BEO assistant. IF a tool returns a menu or specific text, output it EXACTLY as provided. Do not add unverified ingredients."
        }
        
        recent = self.stm.get_context()["recent_messages"]  
        current_messages = [system_msg] + [self._to_groq_format(m) for m in recent]
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
        groq_tools.append({
            "type": "function",
            "function": {
                "name": "search_policy_knowledge",
                "description": "Search Aurelia's internal policy knowledge base using hybrid retrieval.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The policy question."}},
                    "required": ["query"], "additionalProperties": False
                }
            }
        })
        groq_tools.append({
            "type": "function",
            "function": {
                "name": "recall_guest_memory",
                "description": "Recall durable facts about a guest from long-term semantic memory.",
                "parameters": {
                    "type": "object",
                    "properties": {"guest_id": {"type": "string", "description": "e.g. GUEST_VIP_1"}},
                    "required": ["guest_id"], "additionalProperties": False
                }
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
            self.stm.add_message(Message(
                role="user", content=user_prompt,
                metadata=self._metadata_for_user_message(user_prompt)
            ))
            self.stm.add_message(Message(
                role="assistant",
                content=response_message.content or "",
                metadata={"tool_calls": response_message.tool_calls}
            ))
            
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
                    elif func_name == "search_policy_knowledge":
                        result_text = await self._run_rag_search(args.get("query", ""))
                    elif func_name == "recall_guest_memory":
                        result_text = await self._run_memory_recall(args.get("guest_id", ""))
                    else:
                        result = await session.call_tool(func_name, arguments=args)
                        result_text = result.content[0].text
                    
                    print(f"   [Tool Result]: {result_text}")
                    self.stm.add_message(Message(
                        role="tool",
                        content=result_text,
                        metadata={"tool_call_id": tool_call.id, "name": func_name}
                    ))
                    if func_name in ("book_event_room", "confirm_event_booking"):
                        self.stm.update_scratchpad(
                            current_goal=f"Process booking for {args.get('event_id')}",
                            active_sub_goal=f"Executed {func_name}",
                            working_facts={"event_id": args.get("event_id"), "last_tool": func_name}
                        )
                except Exception as e:
                    print(f"   [Tool Error]: {str(e)}")
                    self.stm.add_message(Message(
                        role="tool",
                        content=f"Error: {str(e)}",
                        metadata={"tool_call_id": tool_call.id, "name": func_name}
                    ))

            recent = self.stm.get_context()["recent_messages"]
            final_response = await self.groq_client.chat.completions.create(
                model=MODEL,
                messages=[system_msg] + [self._to_groq_format(m) for m in recent],
                tools=groq_tools,
                temperature=0.2
            )
            final_text = (final_response.choices[0].message.content or "").strip()
            if not final_text:
                tool_results = [m.content for m in recent if getattr(m, "role", None) == "tool" and m.content]
                final_text = self._recover_empty_final_text(user_prompt, tool_results)
                print(f"\n[Agent]: {final_text}")
            else:
                print(f"\n[Agent]: {final_text}")
            self.stm.add_message(Message(role="assistant", content=final_text))
        else:
            self.stm.add_message(Message(
                role="user", content=user_prompt,
                metadata=self._metadata_for_user_message(user_prompt)
            ))
            final_text = (response_message.content or "").strip()
            if not final_text:
                final_text = self._recover_empty_final_text(user_prompt, [])
            print(f"\n[Agent]: {final_text}")
            self.stm.add_message(Message(role="assistant", content=final_text))

        await self._maybe_run_consolidation()

    async def run_fallback_demo(self):
        print("\n" + "="*65)
        print("STEP 1: THE FALLBACK DEMO (Connecting without Elicitation)")
        print("="*65)
        
        env = os.environ.copy()
        env["DEMO_MODE"] = "fallback"
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            env=env,
        )
        
        async with stdio_client(server_params) as (read_stream, write_stream):

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
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            env=env,
        )
        
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
                    "Use recall_guest_memory to check what we have on file for GUEST_VIP_1.",
                    "Just a note for the file: GUEST_VIP_1 (Eleanor Vance) has a severe nut allergy and is vegan. Please always flag this for her events.",
                    "Attempt to book EVT_999 into ROOM_101 with 500 guests. (This should fail due to fire code).",
                    "Run the chain-wide availability audit for 2026-10-31 to find a different room. (Watch for progress delays)",
                    "Draft a custom menu for GUEST_VIP_1.",
                    "Use search_policy_knowledge to find out the fire code capacity policy for the Grand Magnolia Ballroom.",                  
                    "Use search_policy_knowledge to check whether Aurelia is planning to open a new location in Tokyo next year.",

                    # --- BEFORE CHECK ---
                    "Before we authenticate, check the current deposit status for EVT_999.", 
                    
                    "Authenticate as Senior Director using PIN 1234.",
                    "Now that we are authenticated, please explicitly fetch the tool list using get_available_tools.",
                    "Confirm the event booking for EVT_999 to process the deposit.",
                    
                    # --- AFTER CHECK ---
                    "Check the deposit status for EVT_999 one more time to verify the database changed.",
                    "Use recall_guest_memory to check what we have on file for GUEST_VIP_1 now."
               
                ]
                
                for prompt in demo_prompts:
                    await self.chat_with_groq(prompt, session)
                    await asyncio.sleep(1.0)
                    
                    if "Authenticate" in prompt or "authenticated" in prompt:
                         updated_tools = await session.list_tools()
                         self.tools_available = updated_tools.tools

                # Final consolidation pass
                print("\n   [Memory] Running final consolidation pass...")
                await self.consolidator.run_consolidation_pass()

                print("\n" + "="*65)
                print("MEMORY SUBSYSTEM SUMMARY")
                print("="*65)
                print(f"Short-term buffer size now: {len(self.stm.buffer)}")
                print(f"Episodic entries recorded: {len(self.episodic_store.get_entries())}")
                print("Router decision log:")
                for entry in self.memory_router.get_decision_log():
                    print(f"  [{entry['decision'].upper()}] \"{entry['item_preview']}\" - {entry['reason']}")
                print(f"Semantic facts for GUEST_VIP_1: {self.semantic_store.get_active_facts('GUEST_VIP_1')}")

async def main():
    agent = BEODemoAgent()
    await agent.run_fallback_demo()
    await agent.run_main_demo()

if __name__ == "__main__":
    asyncio.run(main())