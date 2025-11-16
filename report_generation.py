import os
import argparse
import base64
import ollama
import datetime
import json
import re
import math
import requests
import streamlit as st
st.cache_data.clear()
st.cache_resource.clear()

from io import BytesIO
from typing import List, Any, Dict
from PIL import Image

# LangChain imports
from langchain_community.docstore.document import Document
from langchain_classic.agents import Tool

# Langsmith
from langsmith import Client
from langsmith.run_helpers import traceable

# Retrievers
from radrag.retrievers.rexgradient_image_to_text import RexGradientRetriever
from radrag.retrievers.multicare_text_to_text import MedicalMulticareRetriever
from radrag.retrievers.medgemma_image_to_text import generate_initial_diagnosis

# Tools
from radrag.tools.generate_pdf import generate_pdf_report
from radrag.tools.send_email import send_email_with_pdf

# Config
from helpers_config_chat import load_config, OllamaChatSession

import helpers_config_chat as hcc
import importlib
importlib.reload(hcc)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "radrag"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Initialize LangSmith client
try:
    langsmith_client = Client()
except Exception as e:
    st.warning(f"LangSmith not configured: {e}")
    langsmith_client = None

PDF_PATTERNS = [
    r"\bgenerate (a )?pdf( of (this|the) report)?\b",
    r"\bcreate (a )?pdf\b",
    r"\bmake (a )?pdf\b",
]
EMAIL_REGEX = re.compile(
    r"\bsend (an )?email to\s+([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b",
    re.IGNORECASE,
)


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_image_for_ollama(image_path: str) -> str:
    """Load image and convert to base64 for Ollama."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def is_pdf_request(text: str) -> bool:
    t = text.strip().lower()
    return any(re.search(p, t, flags=re.IGNORECASE) for p in PDF_PATTERNS)


def parse_email_request(text: str) -> str | None:
    m = EMAIL_REGEX.search(text.strip())
    return m.group(2) if m else None


def fetch_wikipedia_summary(
    query: str,
    max_chars: int = 1500,
    lang: str = "en",
) -> str:
    """Fetch a short summary from Wikipedia for the given query."""
    if not query.strip():
        return ""

    try:
        search_text = query.strip().splitlines()[0]

        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": search_text,
            "format": "json",
            "srlimit": 1,
        }
        search_resp = requests.get(
            search_url,
            params=search_params,
            timeout=5,
            headers={"User-Agent": "RadRAG/0.1 (contact: you@example.com)"},
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
        matches = search_data.get("query", {}).get("search", [])

        if not matches:
            return ""

        title = matches[0].get("title")
        if not title:
            return ""

        title_slug = title.replace(" ", "_")
        summary_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title_slug}"
        summary_resp = requests.get(
            summary_url,
            timeout=5,
            headers={"User-Agent": "RadRAG/0.1 (contact: you@example.com)"},
        )
        if summary_resp.status_code != 200:
            return ""

        summary_data = summary_resp.json()
        summary = summary_data.get("extract", "") or ""
        return summary[:max_chars].strip()

    except Exception:
        return ""


def fetch_pubmed_summaries(
    query: str,
    max_results: int = 3,
    email: str | None = None,
    api_key: str | None = None,
    max_chars: int = 4000,
) -> str:
    """Fetch a few PubMed abstracts using NCBI E-utilities."""
    if not query.strip():
        return ""

    email = email or os.environ.get("NCBI_EMAIL")
    api_key = api_key or os.environ.get("NCBI_API_KEY")

    try:
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        esearch_params = {
            "db": "pubmed",
            "term": query.strip(),
            "retmax": str(max_results),
            "retmode": "json",
        }
        if email:
            esearch_params["email"] = email
        if api_key:
            esearch_params["api_key"] = api_key

        r = requests.get(
            esearch_url,
            params=esearch_params,
            timeout=8,
            headers={"User-Agent": "RadRAG/0.1 (contact: you@example.com)"},
        )
        r.raise_for_status()
        data = r.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return ""

        ids = ",".join(id_list)

        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        efetch_params = {
            "db": "pubmed",
            "id": ids,
            "rettype": "abstract",
            "retmode": "text",
        }
        if email:
            efetch_params["email"] = email
        if api_key:
            efetch_params["api_key"] = api_key

        r2 = requests.get(
            efetch_url,
            params=efetch_params,
            timeout=12,
            headers={"User-Agent": "RadRAG/0.1 (contact: you@example.com)"},
        )
        r2.raise_for_status()
        text = r2.text.strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", text)
        return cleaned[:max_chars].strip()

    except Exception:
        return ""


@traceable(name="generate_final_report")
def generate_final_report(
    image_base64: str,
    user_prompt: str,
    isd: str,
    isd_rex: str,
    similar_cases: List[Dict[str, Any]],
    external_context: str = "",
    model: str = "amsaravi/medgemma-4b-it:q8"
) -> str:
    """
    Generate detailed medical report using final LLM.
    Sections are included only if their corresponding inputs are non-empty.
    Streamed token-by-token to Streamlit.
    """
    # ---------- Build similar cases text ----------
    cases_text = ""
    for similar_case in similar_cases:
        mc_case_id = similar_case.get('case_id', 'PMC6666666_69')
        mc_patient_age = str(similar_case.get('age', 'Unknown'))
        mc_patient_sex = similar_case.get('gender', 'Unknown')
        mc_text = similar_case.get('text', 'Not available')
        
        case_text = (
            f"Case-ID: {mc_case_id}\n"
            f"Patient-Age: {mc_patient_age}\n"
            f"Patient-Sex: {mc_patient_sex}\n"
            f"Case-Report: {mc_text}\n\n"
        )
        cases_text += case_text

    # ---------- Build prompt in pieces ----------
    blocks: List[str] = []

    # 1) Instructions + user prompt (always present)
    blocks.append(f"INSTRUCTIONS:\n\n{user_prompt}")

    # 2) ISD (VLM-based) – only if non-empty
    if isd and str(isd).strip():
        blocks.append(f"INITIAL ASSESSMENT (ISD):\n{isd}")

    # 3) Image-based reference (RexGradient) – only if non-empty
    if isd_rex and str(isd_rex).strip():
        blocks.append(f"IMAGE-BASED REFERENCE (RexGradient):\n{isd_rex}")

    # 4) Reference cases – only if we have any
    if cases_text.strip():
        blocks.append(f"REFERENCE CASES (MultiCare similar cases):\n{cases_text}")

    # 5) External context – only if provided
    if external_context and external_context.strip():
        blocks.append(f"EXTERNAL CONTEXT:\n{external_context}")

    # 6) Final instruction block – always present
    blocks.append(
        "Based on the chest X-ray image and all provided information above, "
        "generate a structured radiology report with the following sections:\n\n"
        "1. TECHNIQUE: Describe the imaging technique\n"
        "2. FINDINGS: Detailed description of radiographic findings\n"
        "3. IMPRESSION: Summary diagnosis and key findings\n"
        "4. RECOMMENDATIONS: Follow-up care and additional studies if needed\n\n"
        "Be thorough, professional, and clinically accurate. Clearly separate what is "
        "directly supported by the image vs. what is contextual evidence from the literature."
    )

    prompt = "\n\n".join(blocks)

    print(f"Final Prompt:\n{prompt}")

    # Streaming left-to-right into the app
    try:
        print(f"Generating Final Report with {model}...")
        placeholder = st.empty()
        status = st.empty()      # 👈 status line
        report = ""

        for chunk in ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_base64],
            }],
            stream=True,
        ):
            delta = chunk.get("message", {}).get("content", "")
            if not delta:
                continue
            report += delta
            # live update
            placeholder.markdown(report)

        # ✅ Signal that report generation is finished
        status.caption("✅ Report generation finished.")

        print(f"Report: {report}")
        return report.strip()

    except Exception as e:
        st.error(f"Error generating final report: {e}")
        try:
            text_model = "amsaravi/medgemma-4b-it:q8"
            fallback_prompt = (
                f"{prompt}\n\nNote: Image analysis was incorporated from the initial assessment."
            )
            placeholder = st.empty()
            status = st.empty()   # 👈 status line
            report = ""
            for chunk in ollama.chat(
                model=text_model,
                messages=[{
                    'role': 'user',
                    'content': fallback_prompt,
                }],
                stream=True,
            ):
                delta = chunk.get("message", {}).get("content", "")
                if not delta:
                    continue
                report += delta
                placeholder.markdown(report)

            status.caption("✅ Report generation finished (fallback model).")
            return report.strip()
        except Exception as e2:
            return f"Error generating report: {e2}"


class MedicalXRayPipeline:
    
    def __init__(
            self,
            rexgradient_db_path: str,
            rexgradient_db_collection_name: str,
            multicare_db_path: str,
            multicare_docstore_pkl_path: str,
            multicare_db_collection_name: str = "medical_multicare_text",
            multicare_embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
            multicare_rerank_model: str = "amsaravi/medgemma-4b-it:q8",
    ):

        self.mc_retriever = MedicalMulticareRetriever(
            persist_dir=multicare_db_path,
            docstore_pkl=multicare_docstore_pkl_path,
            embedding_model=multicare_embedding_model,
            collection_name=multicare_db_collection_name,
            rerank_model=multicare_rerank_model,
        )

        self.rex_retriever = RexGradientRetriever(
            chroma_dir=rexgradient_db_path, 
            collection=rexgradient_db_collection_name
        )
        
        self.tools = [
            Tool(
                name="GeneratePDF",
                func=lambda x: generate_pdf_report(**json.loads(x)),
                description="Generate a PDF report. Input should be JSON with patient_details, symptoms, final_report, image_path"
            ),
            Tool(
                name="SendEmail",
                func=lambda x: send_email_with_pdf(**json.loads(x)),
                description="Send email with PDF. Input should be JSON with recipient_email, pdf_path, patient_name"
            )
        ]

    @traceable(name="full_pipeline")
    def run_pipeline(
        self,
        query_image_path: str,
        user_prompt: str,
        isd_model: str,
        use_rex: bool = True,
        enable_rag: bool = True,  # global RAG on/off
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline with ISD-Rex + external (Wikipedia/PubMed) support.

        If enable_rag is False:
          - Skip RexGradient retrieval
          - Skip MultiCare similar cases retrieval
          - Skip external context
          - Only use ISD (VLM-based) + final report generation.
        """
        results: Dict[str, Any] = {}
        
        # Step 1: Load image
        query_image_base64 = load_image_for_ollama(query_image_path)
        results['image_loaded'] = True
        
        # Step 2: Generate ISD (VLM-based)
        st.info("Generating initial diagnosis with MedGemma...")
        isd = generate_initial_diagnosis(
            query_image_base64,
            user_prompt,
            isd_model,
        )
        results['initial_diagnosis'] = isd
        st.success("✓ Initial diagnosis (ISD) generated")

        # Display ISD and normalize to text
        st.write("### 🧠 Initial Diagnosis By MedGemma")

        if isinstance(isd, dict):
            isd_text = isd.get("text") or isd.get("answer") or isd.get("prediction") or str(isd)
        elif isinstance(isd, list):
            isd_text = "\n".join(map(str, isd))
        else:
            isd_text = str(isd)

        st.markdown(f"```\n{isd_text}\n```")

        # If ISD is a refusal
        norm_isd = isd_text.lower()

        if "i am unable to provide a diagnosis" in norm_isd or "invalid" in norm_isd:
            st.error("Please choose a relevant Chest X-Ray image.")
            results["error"] = "initial_diagnosis_unavailable"
            st.stop()

        if "diagnos" in norm_isd and any(
            phrase in norm_isd for phrase in [
                "invalid",
                "unable to provide",
                "cannot provide",
                "can't provide",
                "not able to provide",
                "unable to make",
                "cannot make",
                "can't make",
                "not able to make",
            ]
        ):
            st.error("Please choose a relevant Chest X-Ray image.")
            results["error"] = "initial_diagnosis_unavailable"
            st.stop()

        # External context (Wikipedia + PubMed)
        wiki_summary = ""
        pubmed_summary = ""
        external_context = ""

        if enable_rag:
            # If you want external context, uncomment below:
            # st.info("Fetching external context from Wikipedia and PubMed...")
            # wiki_summary = fetch_wikipedia_summary(isd_text, max_chars=1500)
            # pubmed_summary = fetch_pubmed_summaries(isd_text, max_results=3)
            # results["wikipedia_summary"] = wiki_summary
            # results["pubmed_summary"] = pubmed_summary

            external_context_parts = []
            if wiki_summary:
                external_context_parts.append("WIKIPEDIA SUMMARY:\n" + wiki_summary)
            if pubmed_summary:
                external_context_parts.append("PUBMED RESULTS:\n" + pubmed_summary)
            external_context = "\n\n".join(external_context_parts).strip()
        else:
            external_context = ""

        # Step 2B: ISD-Rex (image-based retrieval)
        isd_rex = ""
        if enable_rag and use_rex:
            st.info("Retrieving image-based diagnosis (ISD-Rex) from RexGradient...")
            isd_rex, isd_rex_similarity = self.rex_retriever.retrieve(
                query_image_path,
                return_score=True
            )

            results['isd_rex'] = isd_rex

            st.success(
                f"✓ Image-based diagnosis (ISD-Rex) retrieved | Similarity Score: {isd_rex_similarity * 100.0:.2f}%"
            )

            st.write("### 📝 Similar RexGradient Case")
            if isinstance(isd_rex, dict):
                text = isd_rex.get("text", "")
                st.markdown(f"```\n{text}\n```")
            elif isinstance(isd_rex, str):
                st.markdown(f"```\n{isd_rex}\n```")
            else:
                st.warning("Unexpected ISD-Rex format")
        elif not enable_rag:
            isd_rex = ""
            results['isd_rex'] = ""
        else:
            isd_rex = ""
            results['isd_rex'] = "RexGradient retrieval disabled"

        # Step 3 & 4: similar cases
        similar_docs: List[Dict[str, Any]] = []

        if enable_rag:
            st.info("Retrieving similar cases and reranking them...")
            original_docs = self.mc_retriever.retrieve(isd_text, k=5, rerank_top_n=None)
            similar_docs = self.mc_retriever.retrieve(isd_text, k=5, rerank_top_n=3)
            results['retrieved_cases'] = len(similar_docs)
            results['similar_cases'] = similar_docs

            order_indices = []
            order_case_ids = []

            def doc_key(d):
                return d.get("case_id") or hash(d.get("text"))

            for doc in similar_docs:
                key = doc_key(doc)
                orig_idx = next(
                    (i for i, odoc in enumerate(original_docs)
                     if doc_key(odoc) == key),
                    None
                )
                order_indices.append(orig_idx)
                order_case_ids.append(doc.get("case_id", f"idx{orig_idx}"))

            arrow_idx = " → ".join(str(i) for i in order_indices)
            arrow_ids = " → ".join(str(cid) for cid in order_case_ids)

            st.info(f"🔢 **Reranked order (indices):** {arrow_idx}")
            st.info(f"🆔 **Reranked order (Case IDs):** {arrow_ids}")

            if similar_docs:
                st.success(f"✓ Retrieved and reranked {len(similar_docs)} similar cases")
                st.write("### 🔍 MultiCare Similar Cases")

                for i, doc in enumerate(similar_docs, 1):
                    raw_score = doc.get("score", None)
                    case_id = doc.get("case_id", "Unknown")
                    text = (doc.get("matched_text") or doc.get("text") or "")
                    text_snippet = (text[:250] + "...") if text else ""

                    score = None
                    try:
                        if hasattr(raw_score, "item"):
                            raw_score = raw_score.item()
                        score = float(raw_score)
                        score = max(0.0, min(1.0, score))
                    except Exception:
                        score = None

                    if isinstance(score, float):
                        pct = score * 100.0
                        st.markdown(
                            f"**Case {i}: {case_id}**  \n"
                            f"🧮 **Similarity:** {score:.4f} ({pct:.2f}%)  \n"
                            f"📄 **Excerpt:** {text_snippet}"
                        )
                    else:
                        st.markdown(
                            f"**Case {i}: {case_id}**  \n"
                            f"📄 **Excerpt:** {text_snippet}"
                        )
        else:
            results['retrieved_cases'] = 0
            results['similar_cases'] = []

        # Step 5: final report (streamed)
        st.info("📝 Generating detailed report...")
        final_report = generate_final_report(
            query_image_base64,
            user_prompt,
            isd_text,
            isd_rex,
            similar_docs,
            external_context=external_context,
        )
        results['final_report'] = final_report
        st.success("✓ Detailed report generated")
        
        return results


def main(args):

    st.cache_data.clear()
    st.cache_resource.clear()

    cfg_path = args.config
    cfg = load_config(cfg_path)

    st.set_page_config(page_title="RadRAG: Medical X-Ray Analysis", layout="wide")
    st.title("RadRAG: AI-Powered Radiology Report Generator")

    # ---- Session init ----
    if "pipeline" not in st.session_state:
        try:
            st.session_state.pipeline = MedicalXRayPipeline(
                rexgradient_db_path=cfg["paths"].get("rexgradient_db_path", ""),
                rexgradient_db_collection_name=cfg["paths"].get("rexgradient_collection", "rexgrad_unified"),
                multicare_db_path=cfg["paths"].get("multicare_db_path", "./chroma_db_medical_multicare"),
                multicare_docstore_pkl_path=cfg["paths"].get("multicare_docstore_pkl_path", "./docstore_medical_multicare.pkl"),
                multicare_db_collection_name=cfg["paths"].get("multicare_collection", "medical_multicare_text"),
                multicare_embedding_model=cfg["models"].get("embedding_model", "pritamdeka/S-PubMedBert-MS-MARCO"),
                multicare_rerank_model=cfg["models"].get("chat_model", "llama3.2:3b"),
            )
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")
            st.session_state.pipeline = None

    if "final_report" not in st.session_state:
        st.session_state.final_report = None
    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = None
    if "temp_image_path" not in st.session_state:
        st.session_state.temp_image_path = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_session" not in st.session_state:
        # still created, but we now stream via ollama directly
        st.session_state.chat_session = OllamaChatSession(
            model=cfg["models"].get("chat_model", "amsaravi/medgemma-4b-it:q8"),
            temperature=float(cfg["ollama"].get("temperature", 0.2)),
            top_p=float(cfg["ollama"].get("top_p", 0.9)),
            num_ctx=int(cfg["ollama"].get("num_ctx", 4096)),
        )

    # ---- Upload (optional) ----
    st.subheader("Upload Chest X-Ray (optional)")
    uploaded_file = st.file_uploader("Upload PNG/JPG/JPEG", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded X-Ray", use_container_width=True)
        os.makedirs("temp", exist_ok=True)
        tmp_path = os.path.join("temp", f"xray_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        img.save(tmp_path)
        st.session_state.temp_image_path = tmp_path
        st.success("Image uploaded.")

    # ---- Toggle report generation ----
    report_enabled = st.toggle(
        "Enable structured report generation (RadRAG mode)",
        value=True,
        help=(
            "When enabled, you can generate a full structured radiology report using the image and text above. "
            "When disabled, this app behaves as a normal chat with the Ollama model; uploading an image is optional."
        ),
    )

    # ---- If report generation is enabled, show request box + RAG toggle + button ----
    if report_enabled:
        st.subheader("Enter your request")
        user_prompt = st.text_area(
            "Example:\nGenerate a medical diagnosis for the uploaded X-ray.\n"
            "Name: Elizabeth Hurley, Age: 59, Gender: Female,\n"
            "Indications: Shortness of breath and chest pain.",
            height=120,
        )

        rag_enabled = st.toggle(
            "Enable RAG retrievals (RexGradient, MultiCare similar cases, External context)",
            value=True,
            help=(
                "When enabled, the system will retrieve image-based cases (RexGradient), "
                "text-based similar cases (MultiCare), and external context. "
                "When disabled, only the VLM-based initial diagnosis (ISD) is used to generate the final report."
            ),
        )

        if st.button("🚀 Generate Full Report", type="primary", use_container_width=True):
            if not st.session_state.pipeline:
                st.error("Pipeline is not initialized.")
            elif not st.session_state.temp_image_path:
                st.error("Please upload an X-Ray image first.")
            elif not user_prompt.strip():
                st.error("Please enter a request/prompt with patient details.")
            else:
                st.session_state.final_report = None
                st.session_state.pdf_path = None
                st.session_state.messages = []

                with st.spinner("Running full analysis pipeline..."):
                    try:
                        results = st.session_state.pipeline.run_pipeline(
                            query_image_path=st.session_state.temp_image_path,
                            user_prompt=user_prompt,
                            isd_model=cfg["models"].get("isd_model", "amsaravi/medgemma-4b-it:q8"),
                            use_rex=bool(cfg["rag"].get("use_rex", False)) and rag_enabled,
                            enable_rag=rag_enabled,
                        )
                        st.session_state.final_report = results.get("final_report")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"### 🩻 Final Radiology Report\n\n{st.session_state.final_report}"
                        })
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Report generated successfully! You can chat freely now, or say “Generate a PDF of this report” / “Send an email to you@host.com”."
                        })
                        st.success("Report Generated!")
                    except Exception as e:
                        st.error(f"Pipeline failed: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, the pipeline failed: {e}"})
    else:
        # When report generation is disabled, we don't need the text box here.
        user_prompt = ""
        st.info(
            "Structured report generation is currently **disabled**. "
            "You can still upload an image (optional) and use the free chat below."
        )

    # ---- Single-column layout: Report (if any) + Chat below ----
    st.header("Generated Report")
    if report_enabled and st.session_state.final_report:
        st.markdown(st.session_state.final_report)
    elif report_enabled:
        st.info("Upload an image, enter your prompt, and click **Generate Full Report**.")
    else:
        st.info("Report generation is disabled. Use the chat below for free-form Q&A.")

    st.header("Actions / Chat")

    # Show entire chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input is ALWAYS rendered last (bottom)
    prompt = st.chat_input(
        "💬 Ask follow-up questions, or say 'Generate a PDF of this report' / 'Send an email to you@host.com'"
    )

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        recipient_email = parse_email_request(prompt)
        name_match = re.search(r"\bName\s*:\s*([^\n,]+)", user_prompt or "", flags=re.IGNORECASE)
        patient_name_for_email = (name_match.group(1).strip() if name_match else "Patient")

        response_text = None

        # A) PDF intent
        if is_pdf_request(prompt):
            with st.chat_message("assistant"):
                if not st.session_state.final_report:
                    response_text = "Generate a report first."
                    st.warning(response_text)
                elif not st.session_state.temp_image_path:
                    response_text = "Please upload an X-Ray image first."
                    st.warning(response_text)
                else:
                    with st.spinner("Generating PDF…"):
                        try:
                            pdf_path = generate_pdf_report(
                                patient_details=user_prompt,
                                symptoms="",
                                final_report=st.session_state.final_report,
                                image_path=st.session_state.temp_image_path,
                            )
                            st.session_state.pdf_path = pdf_path
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Report PDF",
                                    data=f,
                                    file_name=os.path.basename(pdf_path),
                                    mime="application/pdf",
                                )
                            response_text = "PDF report created."
                            st.success(response_text)
                        except Exception as e:
                            response_text = f"Failed to generate PDF: {e}"
                            st.error(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        # B) Email intent
        elif recipient_email:
            with st.chat_message("assistant"):
                if not st.session_state.pdf_path:
                    response_text = "Please generate a PDF first."
                    st.warning(response_text)
                else:
                    sender_email = os.environ.get("SENDER_EMAIL")
                    sender_password = os.environ.get("SENDER_PASSWORD")
                    if not sender_email or not sender_password:
                        response_text = "Email creds not set (SENDER_EMAIL, SENDER_PASSWORD)."
                        st.error(response_text)
                    else:
                        result = send_email_with_pdf(
                            recipient_email=recipient_email,
                            pdf_path=st.session_state.pdf_path,
                            patient_name=patient_name_for_email,
                            sender_email=sender_email,
                            sender_password=sender_password,
                        )
                        response_text = result
                        (st.success if result.lower().startswith("email sent") else st.error)(result)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        # C) Otherwise → Normal chat (streamed) using Ollama ONLY
        else:
            # 👇 User message is already shown above; now render assistant reply
            with st.chat_message("assistant"):
                try:
                    def _build_history(max_turns: int = 20) -> list[dict]:
                        msgs = st.session_state.messages[-max_turns:]
                        history = []
                        for m in msgs:
                            role = m.get("role", "user")
                            content = str(m.get("content", ""))
                            if content.strip():
                                history.append({"role": role, "content": content})
                        return history

                    history = _build_history(max_turns=20)

                    # In free-chat mode (report disabled), pass image if available
                    if not report_enabled and st.session_state.temp_image_path:
                        img_b64 = load_image_for_ollama(st.session_state.temp_image_path)
                        # attach image to last user message if possible
                        if history and history[-1]["role"] == "user":
                            history[-1]["images"] = [img_b64]
                        else:
                            history.append({
                                "role": "user",
                                "content": prompt,
                                "images": [img_b64],
                            })

                    model_name = cfg["models"].get("chat_model", "amsaravi/medgemma-4b-it:q8")

                    # 🔄 Local streaming UI: no global dark overlay
                    text_placeholder = st.empty()   # where the assistant text streams
                    status_placeholder = st.empty() # small status line

                    # Show a "spinning" style message (no real spinner overlay)
                    status_placeholder.markdown("🌀 **Generating...**")

                    reply = ""

                    for chunk in ollama.chat(
                        model=model_name,
                        messages=history,
                        stream=True,
                    ):
                        delta = chunk.get("message", {}).get("content", "")
                        if not delta:
                            continue
                        reply += delta
                        # Live left-to-right update
                        text_placeholder.markdown(reply)

                    # ✅ Done signal (no blackout)
                    status_placeholder.markdown("✅ **Response complete.**")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply.strip()}
                    )

                except Exception as e:
                    msg = f"Ollama chat failed: {e}"
                    st.error(msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": msg}
                    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/reportgen_config.yaml",
        help="Path to YAML configuration file",
    )
    args, _ = parser.parse_known_args()
    main(args)