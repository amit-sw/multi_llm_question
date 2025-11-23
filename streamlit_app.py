import streamlit as st
import asyncio
import threading
import queue

from openai import AsyncOpenAI
from openai.types import responses as openai_responses
import google.generativeai as genai

st.set_page_config(page_title="Two LLM Comparator", layout="wide")


# ------------------------------------------------------------
# Gemini client (sync)
# ------------------------------------------------------------
@st.cache_resource
def get_gemini_model():
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    return genai.GenerativeModel("gemini-2.5-flash")


# ------------------------------------------------------------
# OpenAI streaming
# ------------------------------------------------------------
async def stream_openai_response(query, container):
    parallel_api_key = st.secrets["PARALLEL_API_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]

    tools = [
        openai_responses.tool_param.Mcp(
            server_label="parallel_web_search",
            server_url="https://search-mcp.parallel.ai/mcp",
            headers={"Authorization": "Bearer " + parallel_api_key},
            type="mcp",
            require_approval="never",
        )
    ]

    client = AsyncOpenAI(api_key=openai_api_key)
    text = ""

    stream = await client.responses.create(
        model="gpt-5-nano",
        input=query,
        tools=tools,
        stream=True,
    )

    async for event in stream:
        if event.type == "response.output_text.delta":
            delta = event.delta or ""
            text += delta
            container.markdown(text)


# ------------------------------------------------------------
# Gemini streaming (background thread → queue)
# ------------------------------------------------------------
def gemini_worker(query, model, out_queue):
    try:
        resp = model.generate_content(
            query,
            generation_config={"max_output_tokens": 2048},
            stream=True,
        )

        for chunk in resp:
            if hasattr(chunk, "text") and chunk.text:
                out_queue.put(chunk.text)

    finally:
        out_queue.put(None)   # sentinel for "done"


# ------------------------------------------------------------
# Parallel orchestrator
# ------------------------------------------------------------
async def run_parallel(question, gemini_model, openai_box, gemini_box):

    # thread-safe queue for Gemini tokens
    q = queue.Queue()

    # Launch Gemini thread
    thread = threading.Thread(
        target=gemini_worker,
        args=(question, gemini_model, q),
        daemon=True,
    )
    thread.start()

    gemini_text = ""

    async def pump_gemini():
        """Periodically pull from thread queue and update UI."""
        nonlocal gemini_text

        while True:
            try:
                chunk = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            if chunk is None:  # worker finished
                break

            gemini_text += chunk
            gemini_box.markdown(gemini_text)

    # Run OpenAI + Gemini UI pump in parallel
    await asyncio.gather(
        stream_openai_response(question, openai_box),
        pump_gemini(),
    )

    thread.join()


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
def main():
    st.title("Ask OpenAI and Gemini")

    with st.form("qform"):
        question = st.text_area("Your question", height=120)
        submitted = st.form_submit_button("Submit")

    if not submitted or not question.strip():
        return

    gemini_model = get_gemini_model()

    with st.spinner("Streaming...", show_time=True):
        col_left, col_right = st.columns(2)
        col_left.subheader("🤖 OpenAI")
        col_right.subheader("✨ Gemini")

        openai_box = col_left.empty()
        gemini_box = col_right.empty()
        asyncio.run(
            run_parallel(question, gemini_model, openai_box, gemini_box)
        )


if __name__ == "__main__":
    main()
