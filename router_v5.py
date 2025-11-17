import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = "Radiology RAG Assistant"
import re
import torch
import streamlit as st
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from PIL import Image
import numpy as np
from typing import List, Dict, Any


from langchain_core.runnables import RunnableLambda
from langsmith import traceable

from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.document_loaders import WebBaseLoader

from langchain_community.tools import DuckDuckGoSearchResults

from langchain_community.tools import DuckDuckGoSearchRun
import json
from radrag.tools.send_email import send_email_with_pdf

def load_duckduckgo_search_tool():
    return DuckDuckGoSearchRun() 


duckduckgo_search_tool = load_duckduckgo_search_tool()
import json
from radrag.retrievers.rexgradient_text_to_image import UnifiedRadiologyRetriever
from radrag.retrievers.textbook_retriever import TextbookQARetriever # <--- USE CORRECT PATH

import random



import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
TEXTBOOK_CHROMA_PATH = "/ssd_scratch/cvit/saket/textbook_rag/chroma_db_text"
TEXTBOOK_DOCSTORE_PATH = "/ssd_scratch/cvit/saket/textbook_rag/docstore.pkl"
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
# =========================
#  HF / CACHE CONFIGURATION
# =========================
HF_CACHE_DIR = "/ssd_scratch/cvit/saket/hf_cache"
HF_TOKEN = ""

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# =========================
#  RETRIEVER & DATA PATHS
# =========================
CHROMA_DIR = "/ssd_scratch/cvit/saket/rexgradient/texts"
IMAGES_ROOT = "/ssd_scratch/cvit/saket/rexgradient_448xx_resized_images"
COLLECTION_NAME = "text"

# =========================
#  NIH JSONL dataset (filtered)
# =========================
NIH_JSONL = "/ssd_scratch/cvit/saket/nih_chest_xrays_filtered.jsonl"

@st.cache_resource
def load_nih_dataset():
    data = []
    try:
        with open(NIH_JSONL, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        st.error(f"Failed loading NIH JSONL: {e}")
    return data

NIH_DATA = load_nih_dataset()


# =========================================================
#  Load retriever + models (cached)
# =========================================================
@st.cache_resource
def load_retriever():
    st.write("Initializing Unified Radiology Retriever...")
    retriever = UnifiedRadiologyRetriever(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        images_root=IMAGES_ROOT,
    )
    st.success("Retriever ready!")
    return retriever

@st.cache_resource
def load_medgemma():
    model_id = "google/medgemma-4b-it"
    st.write("Loading MedGemma model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
        cache_dir=HF_CACHE_DIR,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        token=HF_TOKEN,
        cache_dir=HF_CACHE_DIR,
    )
    st.success("MedGemma loaded!")
    return model, processor


@st.cache_resource
def load_qwen_router():

    model_id = "Qwen/Qwen2.5-7B-Instruct-1M" 
    st.write(f"⏳ Loading Qwen2 router ({model_id})...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE_DIR, use_fast=True)
    
    try:
    
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto", # Uses bfloat16 or float16
            cache_dir=HF_CACHE_DIR,
        )
    except Exception as e:
        st.error(f"Failed to load Qwen model: {e}")
        raise
        
    st.success("✅ Qwen router loaded!")
    return tokenizer, qwen_model


retriever = load_retriever()
med_model, med_processor = load_medgemma()
qwen_tokenizer, qwen_model = load_qwen_router()

@st.cache_resource
def load_textbook_retriever():
    """
    Loads the TextbookQARetriever and caches the object so the expensive models
    are only loaded once.
    """
    try:
        retriever = TextbookQARetriever(
            chroma_path=TEXTBOOK_CHROMA_PATH,
            docstore_path=TEXTBOOK_DOCSTORE_PATH,
            embedding_model_name=EMBEDDING_MODEL_NAME
        )
        return retriever
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize the Textbook Retriever: {e}")
        return None
textbook_retriever = load_textbook_retriever()
# =====================================================
#  UTIL: call MedGemma stateless
# =====================================================
@traceable
def call_med_gemma_stateless(prompt_text: str) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    inputs = med_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(med_model.device)

    with torch.inference_mode():
        generation = med_model.generate(**inputs, max_new_tokens=512, do_sample=False)

    input_len = inputs["input_ids"].shape[-1]
    decoded = med_processor.decode(generation[0][input_len:], skip_special_tokens=True)
    return decoded.strip()

# After calling:
# qwen_tokenizer, qwen_model = load_qwen_router()

def call_qwen_stateless(system_prompt: str, user_prompt: str) -> str:
    """
    Stateless Qwen inference that supports both system and user messages.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    # Apply chat template (Qwen automatically handles system/user formatting)
    input_ids = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(qwen_model.device)

    with torch.inference_mode():
        output_ids = qwen_model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
        )

    generated_tokens = output_ids[0][input_ids.shape[-1]:]

    decoded = qwen_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return decoded




# =====================================================
#  QWEN ROUTER: classify_with_qwen (STATLESS + STRICT)
# =====================================================





@traceable
def classify_with_qwen(user_input: str, chat_history: List[Dict[str, Any]]) -> str:
    del chat_history  


    system_prompt = (
    "You are a strict intent classifier. Your ONLY job is to output EXACTLY one of four labels: "
    "QUIZ_GENERATION, RANDOM_QUIZ, or DIRECT_CHAT or TEXTBOOK_QA.\n"
    "Do not add any other text."
    )


 
    messages = [
        {"role": "system", "content": system_prompt},
        
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "DIRECT_CHAT"},
        
        {"role": "user", "content": "what is pneumonia?"},
        {"role": "assistant", "content": "DIRECT_CHAT"},
        
        {"role": "user", "content": "generate a quiz on the lungs"},
        {"role": "assistant", "content": "QUIZ_GENERATION"},
        
        {"role": "user", "content": "give me 5 multiple choice questions on aortic dissection"},
        {"role": "assistant", "content": "QUIZ_GENERATION"},
        
        {"role": "user", "content": "can you explain this image"},
        {"role": "assistant", "content": "DIRECT_CHAT"},

        {"role": "user", "content": "make me a test"},
        {"role": "assistant", "content": "QUIZ_GENERATION"},

        {"role": "user", "content": "give me a quiz"},
        {"role": "assistant", "content": "RANDOM_QUIZ"},

        {"role": "user", "content": "quiz me"},
        {"role": "assistant", "content": "RANDOM_QUIZ"},

        {"role": "user", "content": "look up the textbook on physics?"},
        {"role": "assistant", "content": "TEXTBOOK_QA"},
        {"role": "user", "content": "lookup the textbook on anatomy?"},
        {"role": "assistant", "content": "TEXTBOOK_QA"},

        {"role": "user", "content": user_input.strip()}
    ]

    input_ids = qwen_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  
        tokenize=True,
        return_tensors="pt"
    ).to(qwen_model.device)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids)
    }

    with torch.inference_mode():
        out_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=4, 
            do_sample=False,
            eos_token_id=qwen_tokenizer.eos_token_id,
            pad_token_id=qwen_tokenizer.eos_token_id
        )


    generated_ids = out_ids[0][input_ids.shape[-1]:]
    
    
    decoded = qwen_tokenizer.decode(generated_ids, skip_special_tokens=True).strip().upper()

    print(f"[QWEN CLASSIFIER] Generated: '{decoded}'")
   
    if "QUIZ_GENERATION" in decoded:
        return "QUIZ_GENERATION"
    if "RANDOM_QUIZ" in decoded:
        return "RANDOM_QUIZ"
    if "TEXTBOOK_QA" in decoded:
        return "TEXTBOOK_QA"
    else:
        return "DIRECT_CHAT"



@traceable
@st.cache_data 
def generate_multimodal_reasoning(image_path: str, report_text: str) -> str:
    """
    Uses MedGemma to analyze an image and its report by structuring the input
    correctly using the processor's chat template.
    """
    if not os.path.exists(image_path):
        return "Error: Image file not found."

    try:
        image = Image.open(image_path)
    except Exception as e:
        return f"Error: Could not open image. {e}"

    # Define the text prompt for the model
    prompt_text = f"""You are an expert radiologist. Your task is to analyze the provided X-ray image.
Use the accompanying report as essential context to guide your analysis.

1.  Read the report to understand the key findings.
2.  Examine the image to locate the visual evidence for each finding mentioned.
3.  Provide a step-by-step breakdown that clearly explains how the text in the report corresponds to what is visible in the image.

**Radiology Report:**
---
{report_text}
---

**Analysis of the Image based on the Report:**
"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]


    inputs = med_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(med_model.device)
    

    # Generate the response
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = med_model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    
    decoded = med_processor.decode(generation[0][input_len:], skip_special_tokens=True)
    return decoded.strip()


# ------------------------------------------------------------------ #
# MAIN FUNCTION
# ------------------------------------------------------------------ #
import os
import re
import json
from typing import Any, Dict, List

import streamlit as st

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader

def execute_fallback_search(
    user_input: str,
    base_context_str: str,
    base_sources_str: str,
    original_docs: List,
) -> Dict[str, Any]:

    st.warning("Textbook content insufficient. Falling back to DuckDuckGo web search & full-page scraping.")

    ddg_api = DuckDuckGoSearchAPIWrapper()
    internet_content_str = ""
    scraped_links_str = ""

    # ---------- 1. Build smart search query ----------
    search_query = re.sub(r'\bfrom textbooks?\b', '', user_input, flags=re.IGNORECASE)
    search_query = re.sub(r'[\"\']', '', search_query).strip()

    if any(kw in search_query.lower() for kw in ["transformer", "latent", "byte", "llm", "blt"]):
        search_query = f"{search_query} BLT Meta AI arXiv GitHub"

    print(f"[DEBUG] Final DuckDuckGo query: '{search_query}'")

    # ---------- 2. Perform DuckDuckGo JSON Search ----------
    try:
        with st.spinner(f"Searching DuckDuckGo for '{search_query}'..."):
            search_results = ddg_api.results(search_query, max_results=8)
        print("[DEBUG] DuckDuckGo JSON results:", search_results)

    except Exception as e:
        st.error(f"DuckDuckGo search failed: {e}")
        print(f"[DEBUG] DuckDuckGo ERROR: {e}")
        search_results = []

    # ---------- 3. No results ----------
    if not search_results:
        internet_content_str = "No relevant internet results found."
        print("[DEBUG] No results returned from DuckDuckGo.")
    else:
        print(f"[DEBUG] Top result: {search_results[0].get('title')} -> {search_results[0].get('link')}")

    # ---------- 4. Extract first 2 links ----------
    links_to_scrape = []
    for res in search_results[:2]:
        if isinstance(res, dict):
            link = res.get("link", "").strip()
            if link.startswith(("http://", "https://")):
                links_to_scrape.append(link)
                print(f"[DEBUG] Will scrape: {link}")

    # ---------- 5. Scrape full pages (but truncated to 1000 chars!) ----------
    for link in links_to_scrape:
        try:
            with st.spinner(f"Scraping full text from {link}..."):
                loader = WebBaseLoader(link)
                docs = loader.load()

                # Combine page content
                page_text = "\n\n".join([d.page_content for d in docs])

                # LIMIT to first 1000 characters
                page_text = page_text[:1000]

                internet_content_str += (
                    f"--- Full Content from {link} (truncated to 1000 chars) ---\n"
                    f"{page_text}\n\n"
                )

                scraped_links_str += f"- {link}\n"
                print(f"[DEBUG] Scraped {len(page_text)} chars from {link} (TRUNCATED)")

        except Exception as e:
            msg = f"Failed to scrape {link}: {e}"
            st.warning(msg)
            internet_content_str += f"--- Failed to scrape {link} (Error: {e}) ---\n"
            print(f"[DEBUG] SCRAPE FAIL: {e}")

    # ---------- 6. Final synthesis ----------
    system_prompt = """
You are a radiology assistant. Combine textbook context and internet context to provide
an accurate medical answer. Avoid hallucination.
"""

    user_prompt = f"""
TEXTBOOK CONTEXT:
{base_context_str}

INTERNET CONTEXT (TRUNCATED):
{internet_content_str}

QUESTION:
{user_input}

ANSWER:
"""

    with st.spinner("Synthesizing final answer from web + textbooks..."):
        final_answer = call_qwen_stateless(system_prompt, user_prompt)

    return {
        "answer": f"{final_answer}\n\n**Sources:**\n{base_sources_str}{scraped_links_str}",
        "retrieved_docs": original_docs,
    }


@traceable
def run_textbook_qa_with_fallback(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    user_input = input_dict["user_input"]

    # ------------------------------------------------------------------ #
    # 1. Textbook unavailable
    # ------------------------------------------------------------------ #
    if textbook_retriever is None:
        return {
            "answer": "Sorry, the textbook knowledge base is currently unavailable.",
            "retrieved_docs": []
        }

    # ------------------------------------------------------------------ #
    # 2. Retrieve from textbooks
    # ------------------------------------------------------------------ #
    with st.spinner("Extracting topic from question..."):
        clean_query = extract_textbook_topic(user_input)

    with st.spinner("Searching radiology textbooks..."):
        retrieved_docs = textbook_retriever.retrieve(clean_query, k=10)

    # ------------------------------------------------------------------ #
    # 3. If no docs -> fallback immediately
    # ------------------------------------------------------------------ #
    if not retrieved_docs:
        st.warning(f"No relevant content found in textbooks for '{clean_query}'.")
        return execute_fallback_search(
            user_input=user_input,
            base_context_str="No textbook content found.",
            base_sources_str="",
            original_docs=[],
        )

    # ------------------------------------------------------------------ #
    # 4. Build textbook context
    # ------------------------------------------------------------------ #
    context_str = ""
    sources_str = ""
    seen = set()

    for doc in retrieved_docs:
        src = os.path.basename(doc.metadata.get("source", "Unknown"))
        pg = doc.metadata.get("page_number", "N/A")

        context_str += f"--- Content from {src}, Page {pg} ---\n{doc.page_content}\n\n"

        key = f"{src} (p. {pg})"
        if key not in seen:
            sources_str += f"- {key}\n"
            seen.add(key)

    # ------------------------------------------------------------------ #
    # 5. SYSTEM PROMPT (strict rules + few shots)
    # ------------------------------------------------------------------ #
    SYSTEM_PROMPT = """
You are a strict rule-following radiology assistant.

RULES:
- If the textbook context is NOT sufficient to answer the user's question,
  respond with EXACTLY: INSUFFICIENT_CONTEXT
- If the context IS sufficient, produce a detailed radiology answer.
- Never output explanations or reasoning steps.
- Your output must be ONLY the final answer or ONLY the token.

### FEW-SHOT EXAMPLES ###

EXAMPLE 1
Context: "This section discusses the tibia and fibula."
Question: "Explain quantum computing."
Answer: INSUFFICIENT_CONTEXT

EXAMPLE 2
Context: "MRI features of meningioma include..."
Question: "Who is Barack Obama?"
Answer: INSUFFICIENT_CONTEXT

EXAMPLE 3
Context: "Liver lesions imaging patterns..."
Question: "What is the capital of France?"
Answer: INSUFFICIENT_CONTEXT

EXAMPLE 4
Context: "CT protocols for abdominal trauma..."
Question: "Describe MRI findings of liver hemangioma."
Answer: A liver hemangioma typically appears as ... [correct radiology answer]

### END OF FEW-SHOT EXAMPLES ###
"""

    # ------------------------------------------------------------------ #
    # 6. USER PROMPT (dynamic)
    # ------------------------------------------------------------------ #
    USER_PROMPT = f"""
Context:
{context_str}

Question:
{user_input}

Answer:
"""

    # ------------------------------------------------------------------ #
    # 7. First LLM pass — evaluate sufficiency
    # ------------------------------------------------------------------ #
    with st.spinner("Checking textbook sufficiency..."):
        initial_answer = call_qwen_stateless(SYSTEM_PROMPT, USER_PROMPT)

    print("[Textbook QA] Initial answer:", initial_answer)

    # ------------------------------------------------------------------ #
    # 8. Trigger fallback properly
    # ------------------------------------------------------------------ #
    if initial_answer.strip() in ["INSUFFICIENT_CONTEXT", "insufficient_context"]:
        return execute_fallback_search(
            user_input=user_input,
            base_context_str=context_str,
            base_sources_str=sources_str,
            original_docs=retrieved_docs,
        )

    # ------------------------------------------------------------------ #
    # 9. Otherwise return the textbook-generated answer
    # ------------------------------------------------------------------ #
    return {
        "answer": f"{initial_answer}\n\n**Sources:**\n{sources_str}",
        "retrieved_docs": retrieved_docs,
    }





# ---------------------------------------------------------------------- #
# Simple textbook-only QA (no fallback) – you already had this
# ---------------------------------------------------------------------- #
@traceable
def run_textbook_qa(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    user_input = input_dict["user_input"]

    if textbook_retriever is None:
        return {
            "answer": "Sorry, the textbook knowledge base is currently unavailable.",
            "retrieved_docs": []
        }

    with st.spinner("Extracting topic from question..."):
        clean_query = extract_textbook_topic(user_input)

    with st.spinner("Searching radiology textbooks..."):
        retrieved_docs = textbook_retriever.retrieve(clean_query, k=10)

    if not retrieved_docs:
        return {
            "answer": "I couldn't find any information on that topic in the loaded textbooks.",
            "retrieved_docs": []
        }

    context_str = ""
    sources_str = ""
    seen = set()
    for doc in retrieved_docs:
        src = os.path.basename(doc.metadata.get("source", "Unknown"))
        pg = doc.metadata.get("page_number", "N/A")
        context_str += f"--- Content from {src}, Page {pg} ---\n{doc.page_content}\n\n"
        key = f"{src} (p. {pg})"
        if key not in seen:
            sources_str += f"- {key}\n"
            seen.add(key)

    prompt = f"""
You are a helpful radiology assistant. Answer the user's question based **only** on the textbook context below.
If the answer is not present, say you could not find it.

**Context from Textbooks:**
{context_str}
**User's Question:** {user_input}
**Answer:**
"""
    with st.spinner("Synthesizing answer from textbook content..."):
        answer = call_med_gemma_stateless(prompt)

    return {
        "answer": f"{answer}\n\n**Sources:**\n{sources_str}",
        "retrieved_docs": retrieved_docs,
    }



@traceable
def run_textbook_qa(input_dict: Dict[str, Any]) -> str:
    user_input = input_dict["user_input"]
    
    if textbook_retriever is None:
        return "Sorry, the textbook knowledge base is currently unavailable."
    with st.spinner("Extracting topic from question..."):
        clean_query = extract_textbook_topic(user_input)
    with st.spinner("Searching radiology textbooks..."):
        retrieved_docs = textbook_retriever.retrieve(clean_query, k=10)

    if not retrieved_docs:
        return "I couldn't find any information on that topic in the loaded textbooks."

    context_str = ""
    sources_str = ""
    seen_sources = set()
    for doc in retrieved_docs:
        source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page_number", "N/A")
        
        context_str += f"--- Content from {source_file}, Page {page_num} ---\n{doc.page_content}\n\n"
        
        source_key = f"{source_file} (p. {page_num})"
        if source_key not in seen_sources:
            sources_str += f"- {source_key}\n"
            seen_sources.add(source_key)

    rag_prompt = f"""
You are a helpful radiology assistant. Answer the user's question based ONLY on the context provided below from radiology textbooks.
If the answer is not in the context, state that you could not find the information.

**Context from Textbooks:**
{context_str}
**User's Question:** {user_input}
**Answer:**
"""

    with st.spinner("Synthesizing answer from textbook content..."):
        final_answer = call_med_gemma_stateless(rag_prompt)
    
    final_answer_with_sources = f"{final_answer}\n\n**Sources:**\n{sources_str}"

    return {
        "answer": final_answer_with_sources,
        "retrieved_docs": retrieved_docs
    }

# =====================================================
#  NEW FUNCTION: Extract Textbook Topic
# =====================================================
@traceable
def extract_textbook_topic(user_input: str) -> str:
    """
    Extracts the core medical topic using Qwen instead of MedGemma.
    Output must be a SHORT, CLEAN topic.
    """
    print(f"[RAG] extract_textbook_topic (Qwen) called for: '{user_input}'")
    system_prompt = (
    "Extract ONLY the medical topic from the user's text.\n"
    "Your task is to REMOVE the question portion and keep the medical topic exactly as written.\n"
    "The topic may include multiple words or clauses.\n\n"
    "Do NOT answer the question.\n"
    "Do NOT explain anything.\n"
    "Do NOT rewrite the topic.\n"
    "Do NOT shorten the topic.\n"
    "Do NOT add extra words.\n"
    "Do NOT produce a complete sentence.\n\n"
    "Output ONLY the remaining medical topic as a clean, searchable phrase."
)

    
    messages = [
        {"role": "system", "content": system_prompt},

        # Few examples for stability
        {"role": "user", "content": "explain from textbooks what is pneumonia"},
        {"role": "assistant", "content": "pneumonia"},

        {"role": "user", "content": "describe the anatomy of the liver from textbooks"},
        {"role": "assistant", "content": "liver anatomy"},

        {"role": "user", "content": "explain PET imaging from textbooks"},
        {"role": "assistant", "content": "PET"},

        # actual user input
        {"role": "user", "content": user_input.strip()}
    ]

    # apply chat template exactly like the classifier
    input_ids = qwen_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(qwen_model.device)

    with torch.inference_mode():
        out = qwen_model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=12,
            do_sample=False,
            eos_token_id=qwen_tokenizer.eos_token_id,
            pad_token_id=qwen_tokenizer.eos_token_id
        )

    generated = out[0][input_ids.shape[-1]:]
    topic = qwen_tokenizer.decode(generated, skip_special_tokens=True).strip()

    # cleanup
    topic = topic.replace("\n", " ").strip()

    # fallback heuristic
    if not topic or len(topic) > 60:
        print("[RAG] Qwen topic extraction unclear, using fallback.")
        fallback = (
            user_input.lower()
            .replace("explain from textbooks", "")
            .replace("explain from textbook", "")
            .replace("explain", "")
            .replace("what is", "")
            .replace("describe", "")
            .replace("how to examine", "")
            .strip()
        )
        return fallback

    print(f"[RAG] Qwen extracted topic: '{topic}'")
    return topic

# =====================================================
#  Topic extraction + general chat
# =====================================================
@traceable
def extract_quiz_topic(user_input: str, chat_history: List[Dict[str, str]]) -> str:
    print(f"[RAG] extract_quiz_topic called for: '{user_input}'")
    messages_for_processing = chat_history[:-1]
    topic_prompt = f"""
Review the user's latest message. They want a quiz.
Extract ONLY the medical topic (e.g., 'pneumonia', 'aortic dissection').
If unclear, say 'general radiology'.

User's message: {user_input}
Topic:
"""
    messages_for_processing.append({"role": "user", "content": [{"type": "text", "text": topic_prompt}]})

    if not messages_for_processing or messages_for_processing[0].get("role") != "system":
        messages_for_processing.insert(0, {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]})

    inputs = med_processor.apply_chat_template(
        messages_for_processing,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(med_model.device)

    with torch.inference_mode():
        generation = med_model.generate(**inputs, max_new_tokens=50, do_sample=False)

    input_len = inputs["input_ids"].shape[-1]
    topic = med_processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()
    return topic



@traceable
def get_general_chat_response(input_dict: Dict[str, Any]) -> str:
    user_input = input_dict["user_input"]
    chat_history = input_dict["chat_history"]

   
    # Read the quiz questions from the session state to build a context block.
    quiz_summary = ""
    if "quiz_cache" in st.session_state and st.session_state.quiz_cache:
        quiz_summary += "CONTEXT: A quiz was just generated. Here are the questions and correct answers:\n"
        for idx, quiz in st.session_state.quiz_cache.items():
            question = quiz.get("question", "N/A")
            correct = quiz.get("correct", "N/A")
            choices = quiz.get("choices", [])
            
            # Try to get the text for the correct answer
            correct_text = ""
            if correct and choices:
                try:
                    correct_idx = ord(correct) - ord('A')
                    if 0 <= correct_idx < len(choices):
                        correct_text = choices[correct_idx]
                except Exception:
                    pass # Ignore parsing errors

            quiz_summary += f"\nQ{idx+1}: {question}\n"
            quiz_summary += f"Answer: {correct} ({correct_text})\n"
    else:
        quiz_summary = "CONTEXT: No quiz is currently active."
    

    messages_for_processing = chat_history[:-1]
    
    # --- UPDATE THE PROMPT ---
    prompt = f"""
You are a helpful medical AI with radiology expertise.

{quiz_summary}

---
Based on the context above, respond to the user's latest message.
If the user asks to "explain the answers" or "why is Q1 correct?", you MUST use the CONTEXT to provide explanations for the quiz questions.
If the user is asking a new, unrelated question, answer it normally.

User's latest message: {user_input}
Assistant:
"""
    # --- END OF PROMPT UPDATE ---

    messages_for_processing.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    if not messages_for_processing or messages_for_processing[0].get("role") != "system":
        messages_for_processing.insert(0, {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical AI."}]})

    inputs = med_processor.apply_chat_template(
        messages_for_processing,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(med_model.device)

    with torch.inference_mode():
        generation = med_model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    input_len = inputs["input_ids"].shape[-1]
    decoded = med_processor.decode(generation[0][input_len:], skip_special_tokens=True)
    return decoded.strip()


# =========================
#  PDF QUIZ GENERATION
# =========================
# =========================
#  PDF QUIZ GENERATION
# =========================

def create_quiz_pdf(_quiz_run_id: int) -> io.BytesIO:
    """
    Generates a PDF of the current quiz from session state.
    Ensures all radiology images are normalized to 8-bit RGB so ReportLab
    renders them correctly without white/cut-off artifacts.
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from PIL import Image
    import numpy as np

    buffer = io.BytesIO()

    # Check if quiz exists
    if 'quiz_cache' not in st.session_state or not st.session_state.quiz_cache:
        st.error("No quiz data to generate PDF.")
        return buffer

    try:
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch
        )

        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Radiology Quiz", styles["h1"]))
        story.append(Spacer(1, 0.25 * inch))

        # Loop through questions
        for i in sorted(st.session_state.quiz_cache.keys()):
            quiz_data = st.session_state.quiz_cache.get(i)
            retrieved_data = (
                st.session_state.results[i]
                if "results" in st.session_state and i < len(st.session_state.results)
                else {}
            )

            if not quiz_data:
                continue

            story.append(Paragraph(f"Question {i+1}", styles["h2"]))
            story.append(Spacer(1, 0.1 * inch))

            # --------------------------------------------------------------
            #  FIXED — NORMALIZE & CONVERT TO 8-BIT RGB BEFORE PDF INSERTION
            # --------------------------------------------------------------
            img_path = retrieved_data.get("image_path")

            if img_path and os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path)

                    # Convert to numpy array
                    np_img = np.array(pil_img)

                    # Normalize dynamic range → 8-bit
                    if np_img.dtype != np.uint8:
                        if np_img.max() > np_img.min():
                            np_img = (np_img - np_img.min()) / (np_img.max() - np_img.min()) * 255
                        np_img = np_img.astype(np.uint8)

                    # Ensure RGB (ReportLab fails on grayscale arrays)
                    if len(np_img.shape) == 2:  # grayscale
                        np_img = np.stack([np_img]*3, axis=-1)

                    pil_norm = Image.fromarray(np_img)

                    # Save to in-memory PNG buffer
                    img_buf = io.BytesIO()
                    pil_norm.save(img_buf, format="PNG")
                    img_buf.seek(0)

                    # Maintain aspect ratio (scaled to fit)
                    aspect = pil_norm.height / float(pil_norm.width or 1)
                    target_w = 3.0 * inch
                    target_h = target_w * aspect

                    # Cap height
                    max_h = 3.0 * inch
                    if target_h > max_h:
                        target_h = max_h
                        target_w = target_h / aspect

                    rl_img = RLImage(img_buf, width=target_w, height=target_h)
                    rl_img.hAlign = "CENTER"
                    story.append(rl_img)
                    story.append(Spacer(1, 0.1 * inch))

                except Exception as e:
                    story.append(Paragraph(f"[Image error: {e}]", styles["Normal"]))

            else:
                story.append(Paragraph("[No image available]", styles["Normal"]))

            # --------------------------------------------------------------
            # QUESTION TEXT
            # --------------------------------------------------------------
            q_text = quiz_data.get("question", "N/A").replace("\n", "<br/>")
            story.append(Paragraph(q_text, styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))

            # --------------------------------------------------------------
            # OPTIONS
            # --------------------------------------------------------------
            choices = quiz_data.get("choices", [])
            correct_letter = quiz_data.get("correct")
            correct_text = "[Answer not specified]"

            for j, choice in enumerate(choices):
                opt_letter = chr(65 + j)
                choice_text = choice or "[Blank Option]"
                story.append(Paragraph(f"{opt_letter}. {choice_text}", styles["Normal"]))

                if opt_letter == correct_letter:
                    correct_text = choice_text

            # Correct answer
            story.append(Spacer(1, 0.15 * inch))
            story.append(
                Paragraph(f"<b>Answer: {correct_letter}. {correct_text}</b>", styles["Normal"])
            )
            story.append(Spacer(1, 0.5 * inch))

        # Build PDF
        doc.build(story)

    except Exception as e:
        st.error(f"Failed to generate PDF: {e}")
        buffer.seek(0)
        buffer.truncate()
        buffer.write(f"Failed to generate PDF: {e}".encode("utf-8"))

    buffer.seek(0)
    return buffer




# =====================================================
#  Topic Expansion (NEW FUNCTION)
# =====================================================
@traceable
def expand_topic_for_retrieval(topic: str) -> str:
    """
    Expands a simple topic into a richer query for better retrieval.
    """
    print(f"[RAG] Expanding topic: '{topic}'")
    
    # This prompt asks the LLM to generate keywords for retrieval
    prompt = f"""
You are a medical expert. A user wants to find radiology cases about a specific topic.
Expand this topic into a rich search query containing key symptoms, findings, and related medical terms.
This query will be used to search a database of radiology reports.

Topic: {topic}

Respond with ONLY the rich search query (a single line of comma-separated terms). Do not add explanation.

Example 1:
Topic: cardiomegaly
Response: cardiomegaly, enlarged heart, increased cardiac silhouette, cardiothoracic ratio, pulmonary venous congestion

Example 2:
Topic: pneumonia
Response: pneumonia, infiltrate, consolidation, lung opacity, effusion, fever, cough

Topic: {topic}
Response:
"""
    
    # Use the existing stateless MedGemma call
    expanded_query = call_med_gemma_stateless(prompt)
    
    # Clean up the response
    expanded_query = expanded_query.strip().replace("\n", ", ")
    print(f"[RAG] Expanded query: '{expanded_query}'")
    

    if not expanded_query: 
        return topic
    if len(expanded_query) > 500:
        safe_cutoff = expanded_query.rfind(',', 0, 500)
        if safe_cutoff != -1:
            return expanded_query[:safe_cutoff].strip()
        
    return expanded_query


def extract_nih_diagnosis(text: str) -> str:
    m = re.search(r"Diagnosis:\s*(.*)", text)
    return m.group(1).strip() if m else "Unknown"


def generate_random_nih_quiz(num_questions: int = 5) -> str:
    if not NIH_DATA:
        return "NIH dataset not available."

    st.session_state.results = []
    st.session_state.quiz_cache = {}
    st.session_state.user_answers = {}
    st.session_state.quiz_run_id += 1
    sample = random.sample(NIH_DATA, min(num_questions, len(NIH_DATA)))
    all_diags = [extract_nih_diagnosis(e["combined"]) for e in NIH_DATA]

    for idx, entry in enumerate(sample):
        img_path = entry["image_path"]
        combined = entry["combined"]
        diag = extract_nih_diagnosis(combined)

        distractor_pool = [d for d in all_diags if d != diag]
        distractors = random.sample(distractor_pool, min(3, len(distractor_pool)))
        while len(distractors) < 3:
            distractors.append("No Finding")

        options = [diag] + distractors
        random.shuffle(options)
        correct_letter = "ABCD"[options.index(diag)]

        st.session_state.results.append({
            "image_path": img_path,
            "text": combined
        })

        raw_text = "Question: What is the most likely diagnosis?\n"
        for i, opt in enumerate(options):
            raw_text += f"{chr(65+i)}. {opt}\n"
        raw_text += f"Answer: {correct_letter}\n"

        st.session_state.quiz_cache[idx] = {
            "question": "What is the most likely diagnosis?",
            "choices": options,
            "correct": correct_letter,
            "raw": raw_text,
            "id": img_path
        }

    return f"Generated {len(st.session_state.quiz_cache)} random NIH quiz questions."


# =========================
#  Parsing helpers
# =========================
def parse_answer_from_quiz(text: str) -> str:
    if not text:
        return None
    m = re.search(r"Answer\s*[:\-]\s*([A-D])", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"(?:Correct\s+answer|Correct)\s*[:\-]?\s*([A-D])", text, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return None

def extract_choices_from_quiz(text: str) -> List[str]:
    if not text:
        return []
    pattern = r"(?m)^[ \t]*([A-D])[\.\)]\s*(.+)$"
    matches = re.findall(pattern, text)
    letter_to_text = {letter.upper(): body.strip() for letter, body in matches}
    return [letter_to_text.get(l, "") for l in "ABCD"]

def remove_answer_from_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"(?i)^\s*(Answer|Correct answer|Correct)[\s:.-]*[A-D]\s*$", "", text, flags=re.MULTILINE)
    return cleaned.strip()


def generate_mcq_from_report(report_text: str) -> str:
    """
    Generates a Multiple Choice Question from a report.
    The correct answer is derived from the report's main finding.
    The incorrect answers (distractors) are generated by the AI and MUST NOT be in the report.
    """
    
    prompt = f"""You are an AI assistant creating a multiple-choice question from a radiology report.

**Your task is to follow these steps:**

1.  **Think Step-by-Step:** First, read the report below to understand it thoroughly. Write down your thought process.
    * Identify the **single most important finding** in the report. This is your "ground truth."
2.  **Create the Question:**
    * **Correct Answer:** Write one answer choice that is this "ground truth" finding.
    * **Incorrect Answers (Distractors):** Create three *other* answer choices. These **must** be plausible-sounding radiological findings, but they **must not** be mentioned in the report. They should be completely made up for this quiz.
3.  **Assemble the Quiz:**
    * The user will only see the image, the report is for your reference to make question. So make the question appropriately
    * Present the question as: "What is the primary finding in this image?"
    * List your one correct answer and three incorrect distractors in a random order (A, B, C, D).
    * State the correct answer letter on the final line.

---
**Example Task:**

**Report:** "Lungs are clear. No active infiltrate or effusion. Mild peribronchial thickening is noted. The heart is normal."

**Required Output:**

**Thought Process:**
1.  **Main Finding:** The report is mostly normal ("lungs clear," "heart normal") but calls out one specific, mild finding: "Mild peribronchial thickening." This is the ground truth.
2.  **Correct Answer:** "Mild peribronchial thickening."
3.  **Distractors (Made Up):** I need three plausible lung findings not in the report. I will make up: "A large pneumothorax," "Lobar pneumonia," and "Pulmonary embolism."

---
**Quiz:**
Question: What is the primary finding in this report?
A. A large pneumothorax
B. Mild peribronchial thickening
C. Lobar pneumonia
D. Pulmonary embolism

Answer: B
---

**Now, apply this logic to the report below.**

**Provided Report:**
---
{report_text}
---

**Your Output:**
"""
    return call_med_gemma_stateless(prompt)

def generate_single_image_quiz(report_text: str) -> str:
    prompt = f"""
You are a radiology educator. Your task is to create one multiple-choice question (MCQ) based on the single radiology report provided.

**Instructions:**
1.  Read the report and identify the single most important finding.
2.  Create a clear question about this main finding (e.g., "What is the primary abnormality?" or "What finding is described in the report?").
3.  Generate four plausible options (A, B, C, D). One must be correct based on the report, and the others must be plausible but incorrect.
4.  The options must be on separate lines, starting with "A.", "B.", "C.", "D."
5.  Crucially, you MUST include the correct answer on its own line at the very end, in the exact format: "Answer: X"

**Radiology Report:**
---
{report_text}
---

**Generated Quiz:**
"""
    return call_med_gemma_stateless(prompt)

# =========================
#  RAG Pipeline
# =========================
@traceable
# =========================
#  RAG Pipeline
# =========================
@traceable
def run_and_store_rag(input_dict: Dict[str, Any]) -> str:
    print(f"[RAG] run_and_store_rag invoked for: '{input_dict['user_input']}'")
    user_input = input_dict["user_input"]
    chat_history = input_dict.get("chat_history", [])
    st.session_state.results = []
    st.session_state.quiz_cache = {}
    st.session_state.user_answers = {}
    st.session_state.quiz_run_id += 1

    # --- STEP 1: Extract the simple topic ---
    with st.spinner("Extracting quiz topic..."):
        topic = extract_quiz_topic(user_input, chat_history)
        print("EXTRACTED TOPIC:",topic)
    if not topic or topic.lower() in ["general", "unclear"]:
        return "Please specify a medical topic (e.g., 'pneumonia', 'stroke')."

    # --- STEP 2: Expand the topic into a rich query (NEW) ---
    with st.spinner(f"Expanding topic '{topic}' for better search..."):
        rich_query = expand_topic_for_retrieval(topic)
    # --- END OF NEW STEP ---

    # --- STEP 3: Retrieve using the new rich query ---
    with st.spinner(f"Retrieving cases for '{topic}'..."):
        # Use the new rich_query here instead of the simple topic
        print("RICH QUERY",rich_query)
        results = retriever.retrieve(rich_query, k=5)
        st.session_state.results = results

    if not results:
        # Update the error message to be more helpful
        return f"No cases found for '{topic}' (searched with: '{rich_query}'). Try another topic."

    # --- (Rest of the function is unchanged) ---
    for idx, r in enumerate(results):
        report_text = r.get("text", "")
        try:
            quiz_text = generate_mcq_from_report(report_text)
        except Exception as e:
            quiz_text = f"Error: {e}"
        
        print("Quiz Text:",quiz_text)
        correct = parse_answer_from_quiz(quiz_text)
        choices = extract_choices_from_quiz(quiz_text)
        visible_question = remove_answer_from_text(quiz_text)

        st.session_state.quiz_cache[idx] = {
            "question": visible_question,
            "choices": choices,
            "correct": correct,
            "raw": quiz_text,
            "id": r.get("id", "")
        }

    # Update the return string to show the user what happened
    return f"Generated {len(results)} quiz questions on **{topic}** (based on query: *{rich_query}*)."


# =========================
#  ROUTER CHAIN (FIXED)
# =========================
def classify_intent_chain(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    user_input = input_dict["user_input"]
    intent = classify_with_qwen(user_input, [])  # Stateless
    print(f"[ROUTER] Classified intent: {intent} for input: '{user_input}'")
    return {
        "intent": intent,
        "user_input": user_input,
        "chat_history": input_dict.get("chat_history", [])
    }

intent_classifier_chain = RunnableLambda(classify_intent_chain)
radiology_rag_runnable = RunnableLambda(run_and_store_rag)
general_chat_runnable = RunnableLambda(get_general_chat_response)
textbook_qa_runnable = RunnableLambda(run_textbook_qa_with_fallback)

def route_based_on_intent(input_dict: Dict[str, Any]) -> str:
    intent = input_dict["intent"]
    user_input = input_dict["user_input"]
    if intent == "TEXTBOOK_QA":
        return textbook_qa_runnable.invoke(input_dict)
    if intent == "QUIZ_GENERATION":
        return radiology_rag_runnable.invoke(input_dict)

    if intent == "RANDOM_QUIZ":
        return generate_random_nih_quiz(num_questions=5)

    return general_chat_runnable.invoke(input_dict)


router_chain = intent_classifier_chain | RunnableLambda(route_based_on_intent)



import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

def send_quiz_email(receiver_email: str, pdf_bytes: bytes, quiz_run_id: int):
    """Send the generated quiz PDF by email."""

    sender = st.secrets["email"]["sender"]
    password = st.secrets["email"]["password"]
    smtp_server = st.secrets["email"]["smtp_server"]
    smtp_port = int(st.secrets["email"]["smtp_port"])

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver_email
    msg["Subject"] = f"Radiology Quiz #{quiz_run_id}"

    # Email body
    msg.attach(MIMEText("Attached is your Radiology Quiz PDF.", "plain"))

    # Attachment
    part = MIMEBase("application", "octet-stream")
    part.set_payload(pdf_bytes)
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f"attachment; filename=radiology_quiz_{quiz_run_id}.pdf"
    )
    msg.attach(part)

    # Send email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver_email, msg.as_string())



# =========================
#  Streamlit UI
# =========================
st.set_page_config(layout="wide")
st.title("Radiology RAG & Quiz Assistant")
st.markdown("Ask to **generate a quiz on [topic]** or chat normally.")

# Session state
for key in ['messages', 'results', 'quiz_cache', 'user_answers', 'download_quizzes']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'messages' else {}
if 'quiz_run_id' not in st.session_state:
    st.session_state['quiz_run_id'] = 0
# Chat history
for msg in st.session_state.messages:
    if msg.get("content") and isinstance(msg["content"], list):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"][0]["text"])



# Input
if prompt := st.chat_input("Type your message..."):
    user_msg = {"role": "user", "content": [{"type": "text", "text": prompt}]}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- MODIFIED RESPONSE HANDLING ---
    # Clear any previously retrieved docs before the new call
    if 'last_retrieved_docs' in st.session_state:
        st.session_state.last_retrieved_docs = None
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = router_chain.invoke({
                "user_input": prompt,
                "chat_history": st.session_state.messages
            })

            # Check if the response is from the textbook QA (a dict) or something else (a string)
            if isinstance(response, dict) and "retrieved_docs" in response:
                answer_text = response["answer"]
                # Store the retrieved docs in session state to be displayed later
                st.session_state.last_retrieved_docs = response["retrieved_docs"]
            else:
                answer_text = response # It's just a string for quizzes or general chat

            assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
            st.session_state.messages.append(assistant_msg)
            st.markdown(answer_text)


if 'last_retrieved_docs' in st.session_state and st.session_state.last_retrieved_docs:
    with st.expander("View Retrieved Textbook Content", expanded=True): # expanded=True makes it open by default
        for i, doc in enumerate(st.session_state.last_retrieved_docs):
            source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
            page_num = doc.metadata.get("page_number", "N/A")
            
            st.markdown(f"---")
            st.markdown(f"**Source Document {i+1}:** `{source_file}`, Page `{page_num}`")
            
            # The key is important to make each text_area unique
            st.text_area(
                label="Content", 
                value=doc.page_content, 
                height=200, 
                disabled=True,
                key=f"retrieved_doc_{i}_{source_file}_{page_num}" 
            )


# Quiz display
if st.session_state.results:
    st.subheader("Quiz Questions")

    quiz_id = st.session_state.quiz_run_id

    for i, r in enumerate(st.session_state.results):
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            img_path = r.get("image_path", "")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    np_img = np.array(img)
                    if np_img.max() > np_img.min():
                        norm = (np_img - np_img.min()) / (np_img.max() - np_img.min()) * 255
                        img = Image.fromarray(norm.astype(np.uint8))
                    st.image(img, caption=f"Case {i+1}")
                except Exception as e:
                    st.warning(f"Image load failed: {e}")
            else:
                st.info("No image available")

        with col2:
            # --- FIX: USE THE UNIQUE QUIZ ID IN THE KEY ---
            st.text_area("Report", r.get("text", ""), height=120, key=f"report_{i}_{quiz_id}")
            
            quiz = st.session_state.quiz_cache.get(i, {})
            if quiz:
                st.markdown(f"**Q{i+1}:** {quiz['question']}")
                choices = quiz.get("choices", [])
                options = [f"{chr(65+j)}. {c}" if c else chr(65+j) for j, c in enumerate(choices)]
                
                if options:
                    # --- FIX: USE THE UNIQUE QUIZ ID IN THE KEY ---
                    selected = st.radio("Your answer", options, key=f"ans_{i}_{quiz_id}")
                    st.session_state.user_answers[str(i)] = chr(65 + options.index(selected))
                else:
                    # This is a fallback, but it should also have a unique key
                    # --- FIX: USE THE UNIQUE QUIZ ID IN THE KEY ---
                    st.radio("Your answer", ["A", "B", "C", "D"], key=f"ans_{i}_{quiz_id}")
            with st.expander("Show AI Reasoning (Image + Report)"):
                with st.spinner("Analyzing image and report..."):
                    reasoning = generate_multimodal_reasoning(
                        image_path=r.get("image_path", ""),
                        report_text=r.get("text", "")
                    )
                    st.info(reasoning)
    # This button logic remains the same as it reads from session_state directly
    if st.button("Submit Quiz"):
        score = sum(1 for i in st.session_state.quiz_cache if st.session_state.user_answers.get(str(i)) == st.session_state.quiz_cache[i].get("correct"))
        st.success(f"Score: {score}/{len(st.session_state.quiz_cache)}")


st.subheader("Download")

# 1. Start with the main chat history
txt = "\n\n".join([f"{m['role'].upper()}: {m['content'][0]['text']}" for m in st.session_state.messages])

# --- THIS IS THE FIX ---
# 2. Check if the quiz cache exists and has data
if "quiz_cache" in st.session_state and st.session_state.quiz_cache:
    
    # 3. Add a clear separator
    txt += "\n\n\n===== FULL QUIZ QUESTIONS (RAW) =====\n"
    
    # 4. Loop through the cached quizzes
    for idx, quiz_data in st.session_state.quiz_cache.items():
        
        # 5. Get the 'raw' text (which includes the hidden answer)
        raw_quiz = quiz_data.get("raw", "Quiz text not found")
        quiz_id = quiz_data.get("id", "N/A")
        
        # 6. Append the formatted quiz data to the text variable
        txt += f"\n--- Quiz for Case {idx+1} (ID: {quiz_id}) ---\n{raw_quiz}\n"
# --- END OF FIX ---

# 7. Update the download button to use the new combined text
st.download_button(
    label="Download Full Chat & Quizzes",  # <-- New label
    data=txt,
    file_name="chat_and_quiz_log.txt",  # <-- New filename
    mime="text/plain"
)

if "quiz_cache" in st.session_state and st.session_state.quiz_cache:
    st.markdown("---") # Visual separator
    
    # Call the cached PDF generation function
    pdf_data = create_quiz_pdf(st.session_state.quiz_run_id)
    
    if pdf_data.getbuffer().nbytes > 0:
        st.download_button(
            label="Download Quiz as PDF (with Images)",
            data=pdf_data,
            file_name=f"radiology_quiz_run_{st.session_state.quiz_run_id}.pdf",
            mime="application/pdf"
        )

    st.subheader("Email This Quiz")

    sender_email = st.text_input("Sender Gmail Address")
    sender_password = st.text_input("Gmail App Password (not your real password!)", type="password")
    recipient_email = st.text_input("Recipient Email")

    if st.button("Email Quiz PDF"):
        # Create the PDF in memory
        pdf_buffer = create_quiz_pdf(st.session_state.quiz_run_id)
        pdf_path = f"quiz_{st.session_state.quiz_run_id}.pdf"

        with open(pdf_path, "wb") as f:
            f.write(pdf_buffer.getvalue())

        patient_name = f"Quiz {st.session_state.quiz_run_id}"

        result = send_email_with_pdf(
            recipient_email=recipient_email,
            pdf_path=pdf_path,
            patient_name=patient_name,
            sender_email=sender_email,
            sender_password=sender_password
        )

        if result.startswith("Email sent"):
            st.success(result)
        else:
            st.error(result)
