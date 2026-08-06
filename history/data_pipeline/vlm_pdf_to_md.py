import os
import time
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import google.generativeai as genai

# 1. Configure your Gemini API Key
# Get a free key at https://aistudio.google.com/ if you haven't already
os.environ["GEMINI_API_KEY"] = "AIzaSyB4SL85FvQYZ3fIBDa5w3hHqFLf1IN52Ak"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Using Gemini 1.5 Flash (fast, handles images excellently, high free tier limits)
# model = genai.GenerativeModel('gemini-1.5-flash')
model = genai.GenerativeModel('gemini-3.5-flash')

# Strict System Prompt to ensure clean Markdown output
VLM_PROMPT = """
You are an expert in OCR and document layout analysis.
Please convert the provided document image into clear, structured Markdown text.
Requirements:
1. Extract all text content and maintain the original logical hierarchy (using Markdown headings, lists, bold text, etc.).
2. If the image contains a table, convert it into Markdown table format.
3. If the image contains infographics (e.g., flowcharts, pie charts, relationship diagrams), convert their core logic and data into structured text descriptions or lists.
4. **Strict Restriction**: Output ONLY the converted Markdown text. Do not include any explanatory filler (e.g., "Sure, here is the result").
"""

def pdf_page_to_image(page) -> Image.Image:
    """Convert a single PDF page to a PIL Image (upscaled for better resolution)."""
    zoom_matrix = fitz.Matrix(2.0, 2.0)  # 2x scaling to improve OCR clarity
    pix = page.get_pixmap(matrix=zoom_matrix)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))

def process_batch():
    source_dir = "../knowledge_base/vlm_visual_pdfs"
    target_dir = "../knowledge_base/raw_files"
    
    if not os.path.exists(source_dir):
        print(f"❌ Source folder not found: {source_dir}")
        return
        
    # Ensure the target directory exists where the .md files will go
    os.makedirs(target_dir, exist_ok=True)
    
    pdf_files = list(Path(source_dir).rglob("*.pdf"))
    total_files = len(pdf_files)
    
    print(f"🚀 Starting batch VLM conversion for {total_files} files...\n")

    for index, pdf_path in enumerate(pdf_files, 1):
        # Create a matching .md filename
        md_filename = pdf_path.stem + ".md"
        output_md_path = os.path.join(target_dir, md_filename)
        
        # Skip if already processed (useful if the script gets interrupted and you need to restart)
        if os.path.exists(output_md_path):
            print(f"⏭️ Skipping [{index}/{total_files}]: {md_filename} already exists.")
            continue
            
        print(f"📄 Processing [{index}/{total_files}]: {pdf_path.name}")
        
        try:
            doc = fitz.open(pdf_path)
            full_markdown = []
            
            for page_num in range(len(doc)):
                print(f"   ⏳ Parsing page {page_num + 1}/{len(doc)}...")
                page = doc[page_num]
                img = pdf_page_to_image(page)
                
                try:
                    # Call VLM
                    response = model.generate_content([VLM_PROMPT, img])
                    page_md = response.text.strip()
                    
                    full_markdown.append(f"\n\n\n\n")
                    full_markdown.append(page_md)
                except Exception as e:
                    print(f"   ❌ Failed to parse page {page_num + 1}: {e}")
                
                # Sleep to respect the free API rate limit (15 Requests Per Minute)
                time.sleep(4.5) 
                
            doc.close()
            
            # Save the compiled Markdown
            with open(output_md_path, 'w', encoding='utf-8') as f:
                f.write("".join(full_markdown))
                
            print(f"   ✅ Saved Markdown to: {output_md_path}\n")
            
        except Exception as e:
            print(f"❌ Failed to process {pdf_path.name} entirely: {e}\n")

    print("🎉 Batch VLM conversion is 100% complete!")
    print(f"All generated Markdown files are securely stored in: {target_dir}")
    print("You can now run your main 'load_data.py' script (with RAG_FORCE_REBUILD=1) to index them!")

if __name__ == "__main__":
    process_batch()