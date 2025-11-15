import os
import argparse
import base64
import ollama
import datetime
import json
import re
import math
import streamlit as st
st.cache_data.clear()
st.cache_resource.clear()

from io import BytesIO
from typing import List, Any, Dict
from PIL import Image

# LangChain imports
from langchain_community.docstore.document import Document
from langchain_classic.agents import Tool

#Langsmith
from langsmith import Client
from langsmith.run_helpers import traceable

#Retrievers
from radrag.retrievers.rexgradient_image_to_text import RexGradientRetriever
from radrag.retrievers.multicare_text_to_text import MedicalMulticareRetriever
from radrag.retrievers.medgemma_image_to_text import generate_initial_diagnosis

#Tools
from radrag.tools.generate_pdf import generate_pdf_report
from radrag.tools.send_email import send_email_with_pdf

#Config
from helpers_config_chat import load_config, OllamaChatSession

import inspect, sys
# st.write("Running:", sys.argv)
# st.write("This file:", __file__)

import helpers_config_chat as hcc
# st.write("helpers_config_chat path:", inspect.getfile(hcc))

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

@traceable(name="generate_final_report")
def generate_final_report(
    image_base64: str,
    user_prompt: str,
    isd: str,
    isd_rex: str,
    similar_cases: List[Document],
    model: str = "amsaravi/medgemma-4b-it:q8"
) -> str:
    """
    Generate detailed medical report using final LLM.
    
    Args:
        image_base64: Base64 encoded X-ray image
        patient_details: Patient information
        symptoms: Patient symptoms
        isd: Initial short diagnosis
        isd_rex: Image-based diagnosis from RexGradient retrieval
        similar_cases: Retrieved similar cases
        model: Ollama model name
    
    Returns:
        Detailed medical report
    """
    cases_text = ""
    # Format similar cases
    for similar_case in similar_cases:
        mc_case_id = similar_case.get('case_id', 'PMC6666666_69')
        mc_patient_age = str(similar_case.get('age', 'Unknown'))
        mc_patient_sex = similar_case.get('gender', 'Unknown')
        mc_text = similar_case.get('text', 'Not available')
        
        case_text = f"Case-ID: {mc_case_id}\nPatient-Age:{mc_patient_age}\nPatient-Sex:{mc_patient_sex}\nCase-Report:{mc_text}\n\n"
        cases_text += case_text
    
    prompt = f"""INSTRUCTIONS:

{user_prompt}

INITIAL ASSESSMENT:
{isd}

IMAGE-BASED REFERENCE:
{isd_rex}

REFERENCE CASES (Similar cases for context):
{cases_text}

Based on the chest X-ray image and all provided information above, generate a structured radiology report with the following sections:

1. TECHNIQUE: Describe the imaging technique
2. FINDINGS: Detailed description of radiographic findings
3. IMPRESSION: Summary diagnosis and key findings
4. RECOMMENDATIONS: Follow-up care and additional studies if needed

Be thorough, professional, and clinically accurate. Consider all the provided context including the RexGradient image match."""
    
    print(f"Final Prompt: {prompt}")

    try:
        print(f"Generating Final Report with {model}...")
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_base64]
            }]
        )
        
        report = response['message']['content'].strip()
        print(f"Report: {report}")
        return report
    
    except Exception as e:
        st.error(f"Error generating final report: {e}")
        try:
            text_model = "amsaravi/medgemma-4b-it:q8"
            response = ollama.chat(
                model=text_model,
                messages=[{
                    'role': 'user',
                    'content': f"{prompt}\n\nNote: Image analysis incorporated from initial assessment."
                }]
            )
            return response['message']['content'].strip()
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

    # @traceable(name="full_pipeline")
    # def run_pipeline(
    #     self,
    #     query_image_path: str,
    #     user_prompt: str,
    #     isd_model: str,
    #     use_rex: bool = True
    # ) -> Dict[str, Any]:
    #     """
    #     Run the complete RAG pipeline with ISD-Rex support.
        
    #     Args:
    #         image_path: Path to X-ray image
    #         patient_details: Patient information
    #         symptoms: Patient symptoms
    #         use_rex: Whether to use RexGradient image retrieval (default: True)
        
    #     Returns:
    #         Dictionary with all results
    #     """
    #     results = {}
        
    #     # Step 1: Load image
    #     query_image_base64 = load_image_for_ollama(query_image_path)
    #     results['image_loaded'] = True
        
    #     # Step 2: Generate ISD (VLM-based)
    #     st.info("Generating initial diagnosis with MedGemma...")
    #     isd = generate_initial_diagnosis(
    #         query_image_base64,
    #         user_prompt,
    #         isd_model,
    #     )
    #     results['initial_diagnosis'] = isd
    #     st.success("✓ Initial diagnosis (ISD) generated")
    #     # ------------------------------------------------------
    #     # NEW: Display the ISD (Initial Diagnosis) in Streamlit
    #     # ------------------------------------------------------
    #     st.write("### 🧠 Initial Diagnosis By MedGemma")

    #     if isinstance(isd, dict):
    #         # If model outputs something structured
    #         text = isd.get("text") or isd.get("answer") or isd.get("prediction") or str(isd)
    #         st.markdown(f"```\n{text}\n```")

    #     elif isinstance(isd, list):
    #         # Sometimes LLMs send a list of messages or choices
    #         st.markdown("```\n" + "\n".join(map(str, isd)) + "\n```")

    #     else:
    #         # Raw string or unknown format
    #         st.markdown(f"```\n{isd}\n```")
        
    #     # Step 2B: Generate ISD-Rex (Image-based retrieval)
    #     isd_rex = ""
    #     if use_rex:
    #         st.info("Retrieving image-based diagnosis (ISD-Rex) from RexGradient...")
    #         isd_rex, isd_rex_similarity = self.rex_retriever.retrieve(
    #             query_image_path,
    #             return_score=True
    #         )

    #         results['isd_rex'] = isd_rex

    #         # Display status
    #         st.success(
    #             f"✓ Image-based diagnosis (ISD-Rex) retrieved | Similarity Score: {isd_rex_similarity * 100.0:.2f}%"
    #         )

    #         # -----------------------------------------
    #         # NEW: DISPLAY THE RETRIEVED TEXT PROPERLY
    #         # -----------------------------------------
    #         st.write("### 📝 Similar RexGradient Case")
    #         if isinstance(isd_rex, dict):
    #             # You can decide what fields you want to display
    #             text = isd_rex.get("text", "")
    #             st.markdown(f"```\n{text}\n```")
    #         elif isinstance(isd_rex, str):
    #             st.markdown(f"```\n{isd_rex}\n```")
    #         else:
    #             st.warning("Unexpected ISD-Rex format")
    #     else:
    #         results['isd_rex'] = "RexGradient retrieval disabled"
        
    #     # Step 3: Retrieve similar cases (using ISD)
    #     # Step 4: Rerank
    #     # st.info("Retrieving similar cases and reranking them...")
    #     # similar_docs = self.mc_retriever.retrieve(isd, k=3, rerank_top_n=0)
    #     # results['retrieved_cases'] = len(similar_docs)
    #     # results['similar_cases'] = similar_docs
    #     # st.success(f"✓ Retrieved and reranked {len(similar_docs)} similar cases")
        
    #     st.info("Retrieving similar cases and reranking them...")
    #     similar_docs = self.mc_retriever.retrieve(isd, k=3, rerank_top_n=0)
    #     results['retrieved_cases'] = len(similar_docs)
    #     results['similar_cases'] = similar_docs

    #     if similar_docs:
    #         st.success(f"✓ Retrieved and reranked {len(similar_docs)} similar cases")

    #         st.write("### 🔍 MultiCare Similar Cases")

    #         for i, doc in enumerate(similar_docs, 1):
    #             raw_score = doc.get("score", None)
    #             case_id = doc.get("case_id", "Unknown")
    #             text = (doc.get("text") or "")
    #             text_snippet = (text[:250] + "...") if text else ""

    #             # Safely coerce to float if possible (handles numpy scalars / 0-D arrays)
    #             score = None
    #             try:
    #                 # If it's a numpy array, take the scalar
    #                 if hasattr(raw_score, "item"):
    #                     raw_score = raw_score.item()
    #                 score = float(raw_score)
    #                 # Clamp to [0, 1] just in case
    #                 score = max(0.0, min(1.0, score))
    #             except Exception:
    #                 score = None

    #             if isinstance(score, float):
    #                 pct = score * 100.0
    #                 st.markdown(
    #                     f"**Case {i}: {case_id}**  \n"
    #                     f"🧮 **Similarity:** {score:.4f} ({pct:.2f}%)  \n"
    #                     f"📄 **Excerpt:** {text_snippet}"
    #                 )
    #             else:
    #                 st.markdown(
    #                     f"**Case {i}: {case_id}**  \n"
    #                     f"📄 **Excerpt:** {text_snippet}"
    #                 )

    #     # Step 5: Generate final report (with ISD + ISD-Rex + Similar Cases)
    #     st.info("📝 Generating detailed report...")
    #     final_report = generate_final_report(
    #         query_image_base64,
    #         user_prompt,
    #         isd,
    #         isd_rex,
    #         similar_docs
    #     )
    #     results['final_report'] = final_report
    #     st.success("✓ Detailed report generated")
        
    #     return results
    @traceable(name="full_pipeline")
    def run_pipeline(
        self,
        query_image_path: str,
        user_prompt: str,
        isd_model: str,
        use_rex: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline with ISD-Rex support.
        
        Args:
            image_path: Path to X-ray image
            patient_details: Patient information
            symptoms: Patient symptoms
            use_rex: Whether to use RexGradient image retrieval (default: True)
        
        Returns:
            Dictionary with all results
        """
        results = {}
        
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

        # ------------------------------------------------------
        # Display the ISD (Initial Diagnosis) in Streamlit
        # and normalize to a string so we can inspect it
        # ------------------------------------------------------
        st.write("### 🧠 Initial Diagnosis By MedGemma")

        if isinstance(isd, dict):
            isd_text = isd.get("text") or isd.get("answer") or isd.get("prediction") or str(isd)
        elif isinstance(isd, list):
            isd_text = "\n".join(map(str, isd))
        else:
            isd_text = str(isd)

        st.markdown(f"```\n{isd_text}\n```")

        # ------------------------------------------------------
        # NEW: If ISD is a refusal like
        # "I am unable to provide a diagnosis",
        # stop here and ask for a relevant image.
        # ------------------------------------------------------
        norm_isd = isd_text.lower()

        # Exact phrase from your example
        if "i am unable to provide a diagnosis" in norm_isd:
            st.error("Please choose a relevant Chest X-Ray image.")
            results["error"] = "initial_diagnosis_unavailable"
            st.stop()

        # More general safeguard around "diagnos*"
        if "diagnos" in norm_isd and any(
            phrase in norm_isd for phrase in [
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

        # Step 2B: Generate ISD-Rex (Image-based retrieval)
        isd_rex = ""
        if use_rex:
            st.info("Retrieving image-based diagnosis (ISD-Rex) from RexGradient...")
            isd_rex, isd_rex_similarity = self.rex_retriever.retrieve(
                query_image_path,
                return_score=True
            )

            results['isd_rex'] = isd_rex

            # Display status
            st.success(
                f"✓ Image-based diagnosis (ISD-Rex) retrieved | Similarity Score: {isd_rex_similarity * 100.0:.2f}%"
            )

            # -----------------------------------------
            # DISPLAY THE RETRIEVED TEXT PROPERLY
            # -----------------------------------------
            st.write("### 📝 Similar RexGradient Case")
            if isinstance(isd_rex, dict):
                text = isd_rex.get("text", "")
                st.markdown(f"```\n{text}\n```")
            elif isinstance(isd_rex, str):
                st.markdown(f"```\n{isd_rex}\n```")
            else:
                st.warning("Unexpected ISD-Rex format")
        else:
            results['isd_rex'] = "RexGradient retrieval disabled"
        
        # Step 3 & 4: Retrieve similar cases (using ISD) and rerank
        st.info("Retrieving similar cases and reranking them...")
        similar_docs = self.mc_retriever.retrieve(isd_text, k=3, rerank_top_n=0)
        results['retrieved_cases'] = len(similar_docs)
        results['similar_cases'] = similar_docs

        if similar_docs:
            st.success(f"✓ Retrieved and reranked {len(similar_docs)} similar cases")

            st.write("### 🔍 MultiCare Similar Cases")

            for i, doc in enumerate(similar_docs, 1):
                raw_score = doc.get("score", None)
                case_id = doc.get("case_id", "Unknown")
                text = (doc.get("text") or "")
                text_snippet = (text[:250] + "...") if text else ""

                # Safely coerce to float if possible (handles numpy scalars / 0-D arrays)
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

        # Step 5: Generate final report (with ISD + ISD-Rex + Similar Cases)
        st.info("📝 Generating detailed report...")
        final_report = generate_final_report(
            query_image_base64,
            user_prompt,
            isd_text,
            isd_rex,
            similar_docs
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
        # create Ollama-only chat session for post-report conversations
        st.session_state.chat_session = OllamaChatSession(
            model=cfg["models"].get("chat_model", "amsaravi/medgemma-4b-it:q8"),
            temperature=float(cfg["ollama"].get("temperature", 0.2)),
            top_p=float(cfg["ollama"].get("top_p", 0.9)),
            num_ctx=int(cfg["ollama"].get("num_ctx", 4096)),
        )

    st.subheader("(1) Upload Chest X-Ray")
    uploaded_file = st.file_uploader("Upload PNG/JPG/JPEG", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded X-Ray", use_container_width=True)
        os.makedirs("temp", exist_ok=True)
        tmp_path = os.path.join("temp", f"xray_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        img.save(tmp_path)
        st.session_state.temp_image_path = tmp_path
        st.success("Image uploaded.")

    st.subheader("(2) Enter your request (include patient details)")
    user_prompt = st.text_area(
        "Example:\nGenerate a medical diagnosis for the uploaded X-ray.\n"
        "Name: Elizabeth Hurley, Age: 59, Gender: Female,\n"
        "Indications: Shortness of breath and chest pain.",
        height=120,
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
                        use_rex=bool(cfg["rag"].get("use_rex", False)),
                    )
                    st.session_state.final_report = results.get("final_report")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"### 🩻 Final Radiology Report\n\n{st.session_state.final_report}"
                    })
                    # final_report_text = st.session_state.final_report or ""
                    # if final_report_text.strip():
                    #     st.session_state.messages.append({
                    #         "role": "assistant",
                    #         "content": "### 📝 Final Report\n\n" + final_report_text
                    #     })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Report generated successfully! You can chat freely now, or say “Generate a PDF of this report” / “Send an email to you@host.com”."
                    })
                    st.success("Report Generated!")
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, the pipeline failed: {e}"})

    # ---- Layout: Report + Chat ----
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.header("Generated Report")
        if st.session_state.final_report:
            st.markdown(st.session_state.final_report)
        else:
            st.info("Upload an image, enter your prompt, and click **Generate Full Report**.")

    with col2:
        st.header("Actions / Chat")

        # show entire chat history first
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # now place the chat input AFTER the history
        prompt = st.chat_input(
            "💬 Ask follow-up questions or say 'Generate a PDF of this report' / 'Send an email to you@host.com'"
        )

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)


            # intent helpers you already have
            recipient_email = parse_email_request(prompt)

            # quick name extraction for email subject
            name_match = re.search(r"\bName\s*:\s*([^\n,]+)", user_prompt or "", flags=re.IGNORECASE)
            patient_name_for_email = (name_match.group(1).strip() if name_match else "Patient")

            # Branching:
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
                return

            # B) Email intent
            if recipient_email:
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
                return

            # C) Otherwise → Normal chat (NO RAG) using Ollama ONLY
            with st.chat_message("assistant"):
                try:
                    # 🧠 Build a compact history from the last 20 messages, including the final report
                    def _build_history(max_turns: int = 20) -> list[dict]:
                        msgs = st.session_state.messages[-max_turns:]
                        history = []
                        for m in msgs:
                            role = m.get("role", "user")
                            content = str(m.get("content", ""))
                            if content.strip():
                                history.append({"role": role, "content": content})
                        return history

                    # 🧩 Construct conversation context and call Ollama with full history
                    history = _build_history(max_turns=20)
                    reply = st.session_state.chat_session.ask(prompt, history=history)

                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                except Exception as e:
                    msg = f"Ollama chat failed: {e}"
                    st.error(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/Users/namanmishra/Documents/Code/iiith_courses/lma/major_project/Radiology-RAG/configs/reportgen_config.yaml", help="Path to YAML configuration file")
    args, _ = parser.parse_known_args()  # use parse_known_args() so Streamlit CLI flags don't break
    main(args)