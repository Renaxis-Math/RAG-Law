import streamlit as st
import asyncio
import traceback
from dotenv import load_dotenv
from backend import get_workflow
from langchain_core.tracers.context import collect_runs
from langsmith import Client
from helper.entity_checker import check_entities
from langchain_openai import ChatOpenAI

load_dotenv()

def init_client():
    try:
        client = Client()
        print("LangSmith Client Initialized.")
        return client
    except Exception as e:
        st.warning(f"LangSmith init failed: {e}")
        return None

def sidebar():
    with st.sidebar:
        st.title("📜 Insurance Law Chatbot")
        st.markdown(
            """
            This assistant answers questions about the California Insurance Code.
            It uses Retrieval-Augmented Generation over your local PDF collection.

            **Example Queries**  
            - What does the Insurance Code say about cancellation?  
            - Define "insurable interest" under California law.  
            - What are the statutory requirements for claims handling?
            """
        )
        st.markdown("---")
        if st.button("🔄 New Conversation"):
            st.session_state.history = []
            for k in list(st.session_state.keys()):
                if k.startswith("feedback_") or k.startswith("rating_"):
                    del st.session_state[k]
            st.rerun()

def feedback_form(idx, run_id, client):
    key = f"feedback_{idx}"
    if st.session_state.get(key):
        st.success("Thanks for your feedback!")
    else:
        with st.form(f"form_{idx}"):
            ratings = {}
            for cat in ["Accuracy","Relevance","Clarity","Completeness","Satisfaction"]:
                ratings[cat] = st.selectbox(cat, [1,2,3,4,5], format_func=lambda x:"★"*x+"☆"*(5-x), key=f"{cat}_{idx}")
            comment = st.text_area("Comments (optional)", key=f"comment_{idx}")
            if st.form_submit_button("Submit"):
                for cat, val in ratings.items():
                    client.create_feedback(run_id=run_id, key=cat, score=val, value=f"{val} stars")
                if comment:
                    client.create_feedback(run_id=run_id, key="user_comment", comment=comment)
                st.session_state[key] = True
                st.rerun()

async def run_workflow(query: str):
    # Initialize LLM for entity checking
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.0
    )
    
    # Check entities first
    has_california, has_insurance, clarifying_question = check_entities(query, llm)
    
    # If missing required entities, return clarifying question
    if not (has_california and has_insurance):
        return clarifying_question, None
    
    # Proceed with normal workflow if entities are present
    graph = get_workflow().compile()
    final = None
    run_id = None
    with collect_runs() as cb:
        try:
            async for ev in graph.astream({"question": query}, stream_mode="values"):
                final = ev
        except Exception:
            traceback.print_exc()
    if final and "generation" in final:
        answer = final["generation"]
    else:
        answer = "I'm sorry, I couldn't find an answer in the Insurance Code."
    if cb.traced_runs:
        run_id = str(cb.traced_runs[-1].id)
    return answer, run_id

# Initialize
client = init_client()
st.set_page_config(page_title="Insurance Law Chatbot", layout="centered")
st.title("🤖 Insurance Law Chatbot")
sidebar()

if "history" not in st.session_state:
    st.session_state.history = []

# Render chat history
for idx, msg in enumerate(st.session_state.history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
    if msg["role"]=="ai" and msg.get("run_id"):
        feedback_form(idx, msg["run_id"], client)

# Initial greeting
if not st.session_state.history:
    with st.chat_message("ai"):
        st.write("Hello! Ask me anything about the California Insurance Code.")

# User input
user_input = st.chat_input("Your question…")
if user_input:
    st.session_state.history.append({"role":"human","text":user_input})
    with st.chat_message("human"):
        st.markdown(user_input)
    with st.chat_message("ai"):
        placeholder = st.empty()
        placeholder.status("Fetching answer…")
        ans, rid = asyncio.run(run_workflow(user_input))
        placeholder.markdown(ans)
        st.session_state.history.append({"role":"ai","text":ans,"run_id":rid})
        if rid:
            feedback_form(len(st.session_state.history)-1, rid, client)
