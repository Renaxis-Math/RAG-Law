import streamlit as st
from dotenv import load_dotenv
from backend import get_workflow
import asyncio
from langchain_core.tracers.context import collect_runs
from langsmith import Client
import traceback

load_dotenv()

def initialize_client():
    try:
        client = Client()
        print("LangSmith Client Initialized Successfully.")
        return client
    except Exception as e:
        st.warning(f"Could not initialize LangSmith client. Feedback submission may not work. Error: {e}")
        print(f"LangSmith Client Initialization Failed: {e}")
        return None

def setup_sidebar():
    with st.sidebar:
        st.title("CGU Info Chatbot")
        st.markdown("""
        ## CGU Handbook Sources
        This assistant is powered by CGU Student Handbooks from:
        - SCGH  
        - DPE  
        - MPH  
        - CISAT  
        - DBOS  
        - SAH  
        - IMS
        
        **Example Questions:**
        - What are the academic policies at CGU?
        - What registration procedures does CGU follow?
        - How can I contact academic advising?
        """)
        st.markdown("----")
        if st.button("New Conversation 🔄", use_container_width=True):
            st.session_state.chat_history = []
            feedback_keys = [key for key in st.session_state.keys() if "feedback_submitted_" in key or "_rating" in key or "comment_" in key]
            for key in feedback_keys:
                del st.session_state[key]
            st.rerun()

def feedback_form(index, run_id, client):
    feedback_key = f"feedback_submitted_{index}"
    if st.session_state.get(feedback_key, False):
        st.success("Thank you for your feedback!")
    else:
        with st.form(key=f"feedback_form_{index}"):
            ratings = {}
            for category in categories:
                cat_key = category.lower().replace(" ", "_")
                rating = st.selectbox(
                    category,
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: "★" * x + "☆" * (5 - x),
                    key=f"{cat_key}_{index}"
                )
                ratings[category] = rating
            comment = st.text_area("Additional comments (optional)", key=f"comment_{index}")
            submit_button = st.form_submit_button("Submit Feedback")
        if submit_button:
            if any(rating is None for rating in ratings.values()):
                st.error("Please provide a rating for all categories.")
            else:
                for category, rating in ratings.items():
                    cat_key = category.lower().replace(" ", "_")
                    client.create_feedback(
                        run_id=run_id,
                        key=f"{cat_key}_rating",
                        score=rating,
                        value=f"{rating} stars"
                    )
                if comment:
                    client.create_feedback(
                        run_id=run_id,
                        key="user_comment",
                        comment=comment
                    )
                print(f"Feedback submitted for run_id: {run_id}")
                for category, rating in ratings.items():
                    print(f"{category}: {rating} stars")
                if comment:
                    print(f"Comment: {comment}")
                st.session_state[feedback_key] = True
                st.rerun()

async def get_drucker_response_with_run_id(user_input):
    graph = get_workflow().compile()
    final_state = None
    run_id = None
    ai_response_content = "I'm sorry, I couldn't find a good answer. Could you try rephrasing?"
    with collect_runs() as cb:
        try:
            async for event in graph.astream(
                {"question": user_input},
                stream_mode="values",
                config={"tags": ["streamlit_app_call"]}
            ):
                final_state = event
            if final_state and "generation" in final_state:
                ai_response_content = final_state["generation"]
            if cb.traced_runs:
                run_id = str(cb.traced_runs[-1].id)
                print(f"AI response generated with run_id: {run_id}")
            else:
                print("AI response generated without run_id")
        except Exception as e:
            ai_response_content = "An error occurred while processing your request. Please check logs."
            traceback.print_exc()
    return ai_response_content, run_id

client = initialize_client()

st.set_page_config(
    page_title='CGU Info Chatbot',
    page_icon="📚",
    layout="centered"
)

st.title("🤖 CGU Info Chatbot")

categories = ["Accuracy", "Relevance", "Clarity", "Completeness", "Overall Satisfaction"]

setup_sidebar()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for i, message in enumerate(st.session_state.chat_history):
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)
    if role == "ai" and "run_id" in message:
        feedback_form(i, message["run_id"], client)

if not st.session_state.chat_history:
    with st.chat_message("ai"):
        st.write("Hello, I'm your CGU bot. How can I help you today? 😏")

user_query = st.chat_input("Ask anything about CGU...")
if user_query:
    print(f"User query: {user_query}")
    st.session_state.chat_history.append({"role": "human", "content": user_query})
    with st.chat_message("human"):
        st.markdown(user_query)
    with st.chat_message("ai"):
        message_placeholder = st.empty()
        with message_placeholder.status("Fetching your answer..."):
            try:
                ai_response_content, run_id = asyncio.run(get_drucker_response_with_run_id(user_query))
            except Exception as e:
                st.error(f"Error generating response: {e}")
                print(f"Error in asyncio.run(get_drucker_response_with_run_id): {e}")
                ai_response_content = "Sorry, I encountered an error generating the response."
        message_placeholder.markdown(ai_response_content)
        st.session_state.chat_history.append({"role": "ai", "content": ai_response_content, "run_id": run_id})
        if run_id:
            new_message_index = len(st.session_state.chat_history) - 1
            feedback_form(new_message_index, run_id, client)
        else:
            st.warning("Feedback not available for this message (missing run ID).", icon="⚠️")
    print("--- Finished Processing User Query ---")