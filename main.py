'''A streamlit app that takes in a Resume (PDF/Image) and scores it on basis of the JD.'''

print('App Started...')

import streamlit as st

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from langchain.tools import tool
from datetime import datetime

import os
import fitz  # PyMuPDF to count PDF pages
from google.cloud import vision
from google.cloud import storage


from dotenv import load_dotenv
load_dotenv()
# import pypdf2
# from PIL import Image
# import pytesseract
# import io

print("Successfully Imported...")

def setup_gcs_credentials():
    """Set up Google Cloud credentials"""
    credentials_path = "/Users/muskangupta/Documents/M/Code/ResumeScore/autogmailengine-d40bdaa44b76.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

setup_gcs_credentials()

BUCKET_NAME = "test-bucket-resume"
OUTPUT_PREFIX = "vision_output"

print('Google Cloud Vision Successfully Set Up...')

class StructuredResume(BaseModel):
    personal_info: Dict[str, Any]
    experience: List[Dict[str, Any]]
    education: List[Dict[str, Any]]
    skills: List[str]
    certifications: List[str]
    projects: List[Dict[str, Any]] = []
    publications: List[str] = []
    achievements: List[str] = []


class ResumeScore(BaseModel):
    overall_score: float
    technical_skills_score: float
    experience_score: float
    requirements_score: float
    explanation: str


class GraphState(BaseModel):
    input_file: Any = None
    file_type: str = ""
    extracted_text: str = ""
    structured_resume: Optional[StructuredResume] = None
    job_description: str = ""
    final_score: Optional[ResumeScore] = None

print("States Defined...")

#A1
#Extract text from PDFs, Images and Images embedded in PDFs using Google Cloud Vision API.

def upload_to_gcs(local_path: str, bucket_name: str, blob_name: str):
    """Upload file to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"

def download_from_gcs(bucket_name: str, prefix: str, local_dir: str = "output"):
    """Download all JSON result files from GCS output prefix."""
    os.makedirs(local_dir, exist_ok=True)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    texts = []
    for blob in bucket.list_blobs(prefix=prefix):
        local_path = os.path.join(local_dir, os.path.basename(blob.name))
        blob.download_to_filename(local_path)

        import json
        with open(local_path, "r") as f:
            data = json.load(f)

        for response in data["responses"]:
            if "fullTextAnnotation" in response:
                texts.append(response["fullTextAnnotation"]["text"])
    return "\n".join(texts)

def document_extraction_agent(state: GraphState) -> GraphState:
    """
    Extract text from PDFs, Images and Images embedded in PDFs using Google Cloud Vision API.
    Takes GraphState, processes the file, and returns updated GraphState with extracted text.
    """
    try:
        # Set up GCS bucket
        bucket_name = "test-bucket-resume"
        output_prefix = "vision_output"
        
        # Get file from state and create temporary file
        uploaded_file = state.input_file
        file_type = state.file_type
        
        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        # Extract text using the existing function logic
        extracted_text = extract_resume_text(temp_file_path, bucket_name, output_prefix)
        
        # Update state with extracted text
        state.extracted_text = extracted_text
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return state
    
    except Exception as e:
        st.error(f"Error in document extraction: {str(e)}")
        state.extracted_text = ""
        return state
    
def extract_resume_text(file_path: str, bucket_name: str, output_prefix: str = "vision_output"):
    client = vision.ImageAnnotatorClient()
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # IMAGE CASE
    if ext in [".jpg", ".jpeg", ".png", ".tiff"]:
        with open(file_path, "rb") as f:
            content = f.read()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        return response.full_text_annotation.text

    # PDF CASE 
    elif ext == ".pdf":
        # print('yup \n')
        # Count pages
        doc = fitz.open(file_path)
        num_pages = len(doc)

        if num_pages == 1:
        # if 1 == 1:
            with open(file_path, "rb") as f:
                # print('yup \n')
                content = f.read()
            input_config = vision.InputConfig(content=content, mime_type='application/pdf')
            features = [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)]
            request = vision.AnnotateFileRequest(input_config=input_config, features=features, pages=[1])
            response = client.batch_annotate_files(requests=[request])
            print('heyyyyyy', response.responses[0].responses[0].full_text_annotation.text)
            return response.responses[0].responses[0].full_text_annotation.text
        
        else:
            print(f'Processing multi-page PDF with {num_pages} pages')
            blob_name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(file_path)}"
            gcs_uri = upload_to_gcs(file_path, bucket_name, blob_name)
            output_gcs_uri = f"gs://{bucket_name}/{output_prefix}/"

            feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
            gcs_source = vision.GcsSource(uri=gcs_uri)
            input_config = vision.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")
            gcs_destination = vision.GcsDestination(uri=output_gcs_uri)
            output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=2)

            request = vision.AsyncAnnotateFileRequest(
                features=[feature], input_config=input_config, output_config=output_config
            )
            operation = client.async_batch_annotate_files(requests=[request])
            print("Processing PDF with Vision OCR...")
            operation.result(timeout=300)

            extracted_text = download_from_gcs(bucket_name, output_prefix)
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.delete()
                print(f"Cleaned up temporary file: {blob_name}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary file {blob_name}: {e}")
            
            return extracted_text
            # return download_from_gcs(bucket_name, output_prefix)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

#A2
#Structure Raw Text to JSON

def resume_processing_agent(state: GraphState) -> GraphState:
    """
    Process extracted text and structure into JSON format
    """
    try:
        GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
        print('GEMINI_API_KEY =' ,GEMINI_API_KEY)


        llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    #gemini-2.5-pro
                    #gemini-1.5-pro-latest
                    google_api_key=GEMINI_API_KEY,
                    temperature = 0
                )
        
        parser = PydanticOutputParser(pydantic_object=StructuredResume)

        prompt = f"""
        You will be given extracted text from a Resume.
        Your duty is to structure the following resume text into the specified JSON format.
        
        Resume Text:
        {state.extracted_text}
        
        Please extract:
        - Personal information (name, email, phone, location) - put in personal_info as dict
        - Work experience (company, role, duration, responsibilities) - put in experience as list of dicts
        - Education (degree, institution, year) - put in education as list of dicts
        - Skills (technical and soft skills) - put in skills as list of strings
        - Certifications - put in certifications as list of strings
        - Projects (if any) - put in projects as list of dicts
        - Publications (if any) - put in publications as list of strings
        - Achievements (if any) - put in achievements as list of strings
        
        {parser.get_format_instructions()}
        """

        tokens = llm.client.count_tokens(
        model=llm.model,
        contents=[{"role": "model", "parts": [{"text": prompt}]}]
                                        )
        

        if tokens:
                print("Prompt tokens:", tokens.total_tokens)
        else:
            print("Prompt tokens: (could not be counted)")

        print("Prompt tokens:", tokens.total_tokens)

        response = llm.invoke([HumanMessage(content=prompt)])

        tokens = llm.client.count_tokens(
        model=llm.model,
        contents=[{"role": "model", "parts": [{"text": response.content}]}]
                                        )

        print("Response tokens:", tokens.total_tokens)

        print(f"LLM Response for structuring: {response.content}")
        
        structured_data = parser.parse(response.content)
        state.structured_resume = structured_data
        
        return state
    except Exception as e:
        st.error(f"Error in resume processing: {str(e)}")
        return state


#A3
#Score the Resume

@tool
def get_current_date() -> str:
    """Get the current date in a formatted string."""
    return datetime.now().strftime("%d %B %Y")

def resume_scoring_agent(state: GraphState) -> GraphState:
    """
    Score resume against job description
    """
    try:
        GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

        llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    google_api_key=GEMINI_API_KEY,
                    temperature = 0
                )
        parser = PydanticOutputParser(pydantic_object=ResumeScore)

        current_date = get_current_date.invoke({})

        prompt = f"""
        Job Description:
        {state.job_description}
        
        Resume Data:
        {state.structured_resume.model_dump()}

        Today's Date:
        {current_date}
        
        Please score this resume against the job description on a scale of 0-100:

        1. Technical Skills Match (0-30 points) - Evaluate the candidate's resume on the basis of
                - Keyword Matching: Exact matches for technical skills, tools, frameworks, and programming languages
                - Skill Context: Evidence of practical application (not just listing)
                - Technology Variations: Related technologies that demonstrate transferable skills
                - ATS Considerations: Prioritize exact keyword matches and commonly used skill variations

        2. Experience Relevance (0-35 points) - 
            - Industry/Domain Keywords: Presence of industry-specific terms and contexts
            - Role Title Alignment: Job titles that match or relate to target role, also check the seniority level check
            - Responsibility Overlap: Specific duties that align with job requirements
            - Quantifiable Impact: Metrics, percentages, numbers demonstrating results
            - Experience Duration: Total relevant experience vs job requirements 

        3. Requirements Fulfillment (0-30 points) - 
        Systematically check each stated requirement in the job description.
            - Hard Requirements: Education, certifications, years of experience (Exact matches)
            - Must-have Skills: Non-negotiable technical or domain skills
            - Preferred Qualifications: Nice-to-have skills that add value
            - Keyword Density: Frequency and context of requirement-related terms

        4. Soft Skills (0-5 points) - Evaluate the Soft Skills, Leadership skills in alignment with the job description. 

            Note: Calculate candidate's years of experience on the basis of internships, full-time roles or contacts only. 
            Do not consider personal projects or education in work/professional experience. Calculate the experience from present day only- {current_date}
        
        Provide detailed explanation for the scoring in the explanation field.
        Calculate overall_score as the sum of all individual scores.
        
        
        {parser.get_format_instructions()}
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        print(f"LLM Response for scoring: {response.content}")
        
        score_data = parser.parse(response.content)
        state.final_score = score_data
        
        return state
    except Exception as e:
        st.error(f"Error in resume scoring: {str(e)}")
        return state


print('Agents Defined Successfully...')

#Build Graph

def build_graph():
    graph = StateGraph(GraphState)

    # Add nodes
    # graph.add_node("extract_doc", extract_resume_text)
    graph.add_node("extract_doc", document_extraction_agent) 
    graph.add_node("structure_resume", resume_processing_agent)
    graph.add_node("score_resume", resume_scoring_agent)

    # Define flow
    graph.set_entry_point("extract_doc")
    graph.add_edge("extract_doc", "structure_resume")
    graph.add_edge("structure_resume", "score_resume")
    graph.add_edge("score_resume", END)

    return graph


#Main function

if __name__ == '__main__':
    
    st.title("Resume Scoring System")
    st.write("Upload a resume and provide job description for AI-powered scoring")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Resume", 
        type=['pdf', 'jpg', 'jpeg', 'png', 'txt']
    )
    
    # Job description input
    job_description = st.text_area(
        "Job Description", 
        height=200,
        placeholder="Paste the job description here..."
    )

    if uploaded_file and job_description:
        if "prev_file" not in st.session_state or st.session_state.prev_file != uploaded_file.name:
            st.session_state.prev_file = uploaded_file.name
            st.session_state.graph_state = GraphState(
                input_file=uploaded_file,
                file_type=uploaded_file.name.split('.')[-1].lower(),
                job_description=job_description
            )

    
    if st.button("Score Resume") and uploaded_file and job_description:
        st.session_state.clear() 
        
        # Determine file type
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        # Initialize state
        initial_state = GraphState(
            input_file=uploaded_file,
            file_type=file_type,
            job_description=job_description
        )
        
        # Create and run workflow
        workflow = build_graph().compile()

        with st.spinner("Processing resume..."):
            try:
        
                # final_state = workflow.invoke(initial_state)
                final_result = workflow.invoke(initial_state)
                print("Workflow Invoked...")
                print(type(final_result))

                if isinstance(final_result, dict):
                        final_state = GraphState(**final_result)

                else:
                        final_state = final_result
                
                # Display results
                if final_state.final_score:
                    print('Here')
                    st.success("Resume scored successfully!")
                    
                    # col1, col2 = st.columns(2)
                    
                    # with col1:
                        # print("")

                    print('Here')
                    st.metric("Overall Score", f"{final_state.final_score.overall_score}/100")
                    st.metric("Technical Skills", f"{final_state.final_score.technical_skills_score}/30") 
                    st.metric("Experience", f"{final_state.final_score.experience_score}/35")
                    st.metric("Requirements", f"{final_state.final_score.requirements_score}/30")
                    
                    # with col2:
                    st.subheader("Detailed Analysis")
                    st.write(final_state.final_score.explanation)
                    
                    # if st.checkbox("Show Extracted Resume Data"):
                    #         st.json(final_state.structured_resume.model_dump())
                
            except Exception as e:
                st.error(f"Error processing resume: {str(e)}")

    