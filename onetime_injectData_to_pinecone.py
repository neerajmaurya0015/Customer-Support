import fitz
import pytesseract
from PIL import Image
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


pdf_paths = {
    "billing": "Billing-Zoom-Guide.pdf",
    "account": "Account-Zoom-User-Account-Settings.pdf",
    "technical": "technical-issues-Zoom-Trouble-Shooting-Guide.pdf",
    "security": "other-issues-security-onboarding-guide.pdf"
}

all_docs = []
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

for category, path in pdf_paths.items():
    # Load text with PyPDFLoader
    loader = PyPDFLoader(path)
    docs = loader.load()

    # Extract images and text with PyMuPDF
    pdf_doc = fitz.open(path)
    for page_num in range(len(pdf_doc)):
        page_text = docs[page_num].page_content
        print(page_num)

        page = pdf_doc[page_num]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            # Option 1: Use OCR to extract text from images
            image_text = pytesseract.image_to_string(image)

            # Append image-derived text to page content
            if image_text.strip():
                page_text += f"\nImage {img_index+1} content: {image_text}"

        # Update the document's content with augmented text
        docs[page_num].page_content = page_text
        docs[page_num].metadata["category"] = category

    # Chunk the augmented documents
    chunks = splitter.split_documents(docs)
    all_docs.extend(chunks)

    pdf_doc.close()

# print(all_docs[0])



import os
from dotenv import load_dotenv
load_dotenv()
#import pinecone
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "customer-support-rag"


if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)


from sentence_transformers import SentenceTransformer

# Load free local embedding model
emb_model = SentenceTransformer("all-MiniLM-L6-v2") # dimension 384

# Assume all_docs is your list of documents (e.g., from LangChain loader)
# all_docs = [...]  # Your pre-chunked docs with page_content and metadata


vectors = []
for i, doc in enumerate(all_docs):
    chunk = doc.page_content
    embedding = emb_model.encode(chunk).tolist() 
    
    
    metadata = doc.metadata.copy()  # Preserves original metadata incl. 'category'
    metadata['text'] = chunk  
    
    # Unique ID (use source + page if available, or simple index)
    doc_id = f"doc_{i}_{metadata.get('source', 'unknown').replace('/', '_').replace('.', '_')}_{metadata.get('page', i)}"
    print(doc_id)
    vectors.append({
        "id": doc_id,
        "values": embedding,
        "metadata": metadata
    })


batch_size = 100 
for j in range(0, len(vectors), batch_size):
    batch = vectors[j:j + batch_size]
    index.upsert(vectors=batch)
    print(f"Upserted batch {j // batch_size + 1} of {len(vectors) // batch_size + 1}")

print("All documents embedded and stored in Pinecone with metadata.")