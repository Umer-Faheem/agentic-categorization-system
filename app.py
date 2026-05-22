
# ── app.py — Streamlit Chat Interface ──────────────────────────────────
import os, pickle, numpy as np, streamlit as st
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="Agentic Categorization System",
    page_icon="🤖",
    layout="wide",
)

# ── Load models once ────────────────────────────────────────────────────
@st.cache_resource
def load_bundle(name):
    with open(f"models/{name}_model.pkl", "rb") as f:
        return pickle.load(f)

svm_b = load_bundle("svm")
nb_b  = load_bundle("naive_bayes")
lr_b  = load_bundle("logistic_regression")
rf_b  = load_bundle("random_forest")

def _infer(bundle, features):
    X   = np.array([float(x.strip()) for x in features.split(",")]).reshape(1, -1)
    Xs  = bundle["scaler"].transform(X)
    p   = bundle["model"].predict(Xs)[0]
    conf = bundle["model"].predict_proba(Xs)[0].max()
    return f"Predicted: {bundle['labels'][p]} (confidence: {conf:.2%})"

@tool
def classify_with_svm(features: str) -> str:
    """SVM classifier for structured numerical Iris features (sepal_len, sepal_w, petal_len, petal_w)."""
    return _infer(svm_b, features)

@tool
def classify_with_naive_bayes(features: str) -> str:
    """Gaussian Naive Bayes for Iris features. Good probabilistic baseline."""
    return _infer(nb_b, features)

@tool
def classify_with_logistic_regression(features: str) -> str:
    """Logistic Regression for Iris features. Best when interpretability matters."""
    return _infer(lr_b, features)

@tool
def classify_with_random_forest(features: str) -> str:
    """Random Forest for Iris features. Best for non-linear patterns."""
    return _infer(rf_b, features)

ALL_TOOLS = [
    classify_with_svm,
    classify_with_naive_bayes,
    classify_with_logistic_regression,
    classify_with_random_forest,
]

@st.cache_resource
def get_executor():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0,
                   api_key=os.getenv("GROQ_API_KEY"))
    prompt = PromptTemplate.from_template(
        """You are an ML orchestrator.
Tools: {tools}
Format:
Question: {input}
Thought:{agent_scratchpad}
Tool names: [{tool_names}]
Final Answer: ..."""
    )
    agent = create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)
    return AgentExecutor(agent=agent, tools=ALL_TOOLS, verbose=False,
                         max_iterations=5, handle_parsing_errors=True,
                         return_intermediate_steps=True)

executor = get_executor()

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Config")
    st.selectbox("LLM Backend", ["Groq — Llama 3 70B"])
    show_trace = st.toggle("Show Agent Reasoning", value=True)
    st.markdown("---")
    st.caption("Models: SVM · NB · LR · RF")
    st.caption("Dataset: Iris (UCI)")

# ── Header ─────────────────────────────────────────────────────────────
st.title("🤖 Agentic Multi-Model Categorization System")
st.caption("ReAct LLM Agent orchestrating 4 ML classifiers · LangChain · COMSATS Spring 2026")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter iris features e.g. 5.1, 3.5, 1.4, 0.2 or ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning..."):
            result = executor.invoke({"input": prompt})
            answer = result["output"]

        if show_trace and result.get("intermediate_steps"):
            with st.expander("🧠 Agent Reasoning Trace"):
                for step in result["intermediate_steps"]:
                    st.code(str(step), language="text")

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
