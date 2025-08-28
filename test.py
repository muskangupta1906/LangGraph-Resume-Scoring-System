import os
import fitz  # PyMuPDF to count PDF pages
from google.cloud import vision
from google.cloud import storage

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

def extract_resume_text(file_path: str, bucket_name: str, output_prefix: str = "vision_output"):
    client = vision.ImageAnnotatorClient()
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # ---------------- IMAGE CASE ----------------
    if ext in [".jpg", ".jpeg", ".png", ".tiff"]:
        with open(file_path, "rb") as f:
            content = f.read()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        # if response.full_text_annotation:
        #     full_text = response.full_text_annotation.text
        #     structured_text = []
        #     for text in response.text_annotations[1:]:  # Skip first entry (full text)
        #         structured_text.append(text.description)
            # print("Full Text:\n", full_text)
            # print("Structured Text (by block):\n", '\n'.join(structured_text))
            # return full_text  # Maintain original return for compatibility
        return response.full_text_annotation.text

    # ---------------- PDF CASE ----------------
    elif ext == ".pdf":
        # print('yup \n')
        # Count pages
        doc = fitz.open(file_path)
        num_pages = len(doc)

        if num_pages == 1:
            # Inline (single-page PDF)
            with open(file_path, "rb") as f:
                print('yup \n')
                content = f.read()
            input_config = vision.InputConfig(content=content, mime_type='application/pdf')
            features = [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)]
            request = vision.AnnotateFileRequest(input_config=input_config, features=features, pages=[1])
            response = client.batch_annotate_files(requests=[request])
            print(response.responses[0].responses[0].full_text_annotation.text)
            return response.responses[0].responses[0].full_text_annotation.text
        
        else:
            blob_name = os.path.basename(file_path)
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
            # print("Processing PDF with Vision OCR...")
            operation.result(timeout=300)

            # Download results
            return download_from_gcs(bucket_name, output_prefix)
    #         # Multi-page PDF → GCS Async OCR
    #         blob_name = os.path.basename(file_path)
    #         gcs_uri = upload_to_gcs(file_path, bucket_name, blob_name)
    #         output_gcs_uri = f"gs://{bucket_name}/{output_prefix}/"

    #         feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    #         gcs_source = vision.GcsSource(uri=gcs_uri)
    #         input_config = vision.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")
    #         gcs_destination = vision.GcsDestination(uri=output_gcs_uri)
    #         output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=2)

    #         request = vision.AsyncAnnotateFileRequest(
    #             features=[feature], input_config=input_config, output_config=output_config
    #         )
    #         operation = client.async_batch_annotate_files(requests=[request])
    #         print("Processing multi-page PDF...")
    #         operation.result(timeout=300)

    #         # Download results
    #         return download_from_gcs(bucket_name, output_prefix)




#all to gcs
    # elif ext == ".pdf":
    #     print("PDF detected")
    #     # Always go through async OCR with GCS for PDFs
    #     blob_name = os.path.basename(file_path)
    #     gcs_uri = upload_to_gcs(file_path, bucket_name, blob_name)
    #     output_gcs_uri = f"gs://{bucket_name}/{output_prefix}/"

    #     feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    #     gcs_source = vision.GcsSource(uri=gcs_uri)
    #     input_config = vision.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")
    #     gcs_destination = vision.GcsDestination(uri=output_gcs_uri)
    #     output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=2)

    #     request = vision.AsyncAnnotateFileRequest(
    #         features=[feature], input_config=input_config, output_config=output_config
    #     )
    #     operation = client.async_batch_annotate_files(requests=[request])
    #     print("Processing PDF with Vision OCR...")
    #     operation.result(timeout=300)

    #     # Download results
    #     return download_from_gcs(bucket_name, output_prefix)

    else:
        raise ValueError(f"Unsupported file format: {ext}")




bucket_name = "test-bucket-resume"
# Works for images or PDFs (1 page or multiple pages)

# text = extract_resume_text('''/Users/muskangupta/Documents/M/Code/ResumeScore/Screenshot 2025-08-26 at 6.47.21 PM.pdf''', bucket_name)
# print(text)

print('\n crappghtfjhguhjugjhg\n \n')

# text2 = extract_resume_text("/Users/muskangupta/Documents/M/Code/ResumeScore/Screenshot 2025-08-26 at 11.01.33 PM.png", bucket_name)
# for line in text2[0]:
#     print(line)
# print(text2)

# from google.cloud import vision
# from google.cloud import vision
# from google.cloud import storage

# print('hi')
# client = vision.ImageAnnotatorClient()
# print("Auth works ✅")

def setup_gcs_credentials():
    """Set up Google Cloud credentials"""
    credentials_path = "/Users/muskangupta/Documents/M/Code/ResumeScore/autogmailengine-d40bdaa44b76.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

setup_gcs_credentials()


from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    google_api_key=GEMINI_API_KEY,
                    temperature = 0
                )
prompt = "Explain elon musk's company"
response = llm.invoke([HumanMessage(content=prompt)])
print(response)
