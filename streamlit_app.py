
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI



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
        st.subheader("OpenAI (LangChain-OpenAI)")
        with st.spinner("Querying OpenAI model…"):
            # LangChain ChatModels accept either a string or a list of messages
            response_left = openai_llm.invoke(question)
        st.markdown(response_left.content)

    with col_right:
        st.subheader("Gemini (LangChain-Gemini)")
        with st.spinner("Querying Gemini model…"):
            response_right = gemini_llm.invoke(question)
        st.markdown(response_right.content)


if __name__ == "__main__":
    main()
