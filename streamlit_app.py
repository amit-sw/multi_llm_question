
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from google.ai.generativelanguage_v1beta.types import Tool as GenAITool

from openai import OpenAI
from openai.types import responses as openai_responses


# ----- Model factory functions -----
@st.cache_resource
def get_openai_llm():
    """Create a LangChain ChatOpenAI instance using Streamlit secrets."""
    return ChatOpenAI(
        model="gpt-4.1-mini",  # change to whatever OpenAI model you want
        api_key=st.secrets["OPENAI_API_KEY"],
    )


@st.cache_resource
def get_gemini_llm():
    """Create a LangChain ChatGoogleGenerativeAI instance using Streamlit secrets."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # change to your preferred Gemini model
        google_api_key=st.secrets["GOOGLE_API_KEY"],
    )

def get_openai_response(query):
    parallel_api_key=st.secrets['PARALLEL_API_KEY']
    openai_api_key = st.secrets['OPENAI_API_KEY']
    tools = [
        openai_responses.tool_param.Mcp(
            server_label="parallel_web_search",
            server_url="https://search-mcp.parallel.ai/mcp",
            headers={"Authorization": "Bearer " + parallel_api_key},
            type="mcp",
            require_approval="never",
        )
    ]

    response = OpenAI(
        api_key=openai_api_key
    ).responses.create(
        model="gpt-5-nano",
        input=query,
        tools=tools
    )
    return response.output_text


def main():
    st.title("Two-LLM Answer Comparator")
    st.caption(
        "Ask a question once and compare answers from an OpenAI model (left) "
        "and a Gemini model (right)."
    )

    # ----- Question input at the top -----
    with st.form("question_form"):
        question = st.text_area(
            "Your question",
            placeholder="Type your question here…",
            height=120,
        )
        submitted = st.form_submit_button("Get answers")

    if not submitted:
        return

    if not question.strip():
        st.warning("Please enter a question first.")
        return

    # Prepare LLMs
    openai_llm = get_openai_llm()
    gemini_llm = get_gemini_llm()

    # ----- Two vertical containers / columns -----
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("OpenAI")
        with st.spinner("Querying OpenAI model…", show_time=True):
            # LangChain ChatModels accept either a string or a list of messages
            response_left = get_openai_response(question)
        st.markdown(response_left)

    with col_right:
        st.subheader("Gemini")
        search_tool = GenAITool(google_search={})
        with st.spinner("Querying Gemini model…", show_time=True):
            response_right = gemini_llm.invoke(question, tools=[search_tool] )
        st.markdown(response_right.content)


if __name__ == "__main__":
    main()
