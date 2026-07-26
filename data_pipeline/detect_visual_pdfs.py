import os
from pathlib import Path
import fitz  # PyMuPDF

def diagnose_pdfs(directory: str, text_threshold: int = 300, visual_page_ratio: float = 0.3):
    """
    Scans PDFs in the directory to determine if they need VLM multimodal conversion
    based on text density.
    
    Parameters:
    - text_threshold: Pages with fewer characters than this are considered "visual" or "blank" (default: 300).
    - visual_page_ratio: If the ratio of "visual pages" exceeds this threshold, the PDF needs VLM processing (default: 30%).
    """
    print(f"🔍 Scanning directory: {directory}\n")
    
    vlm_candidates = []
    healthy_pdfs = []
    failed_pdfs = []

    # Recursively find all PDF files in the directory
    pdf_files = list(Path(directory).rglob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in the specified directory.")
        return

    for filepath in pdf_files:
        try:
            doc = fitz.open(filepath)
            total_pages = len(doc)
            
            if total_pages == 0:
                doc.close()
                continue
                
            total_chars = 0
            low_text_pages = 0

            # Analyze text density page by page
            for page in doc:
                text = page.get_text().strip()
                char_count = len(text)
                total_chars += char_count
                
                # If a page yields very few characters, it's likely an image, scan, or broken layout
                if char_count < text_threshold:
                    low_text_pages += 1

            avg_chars_per_page = total_chars // total_pages
            ratio = low_text_pages / total_pages

            # Core decision logic
            if ratio >= visual_page_ratio or avg_chars_per_page < 150:
                vlm_candidates.append({
                    "name": filepath.name,
                    "pages": total_pages,
                    "avg_chars": avg_chars_per_page,
                    "visual_ratio": ratio
                })
            else:
                healthy_pdfs.append(filepath.name)

            doc.close()
        except Exception as e:
            failed_pdfs.append(f"{filepath.name} (Error: {e})")

    # ================= Print Diagnostic Report =================
    print("=" * 60)
    print(f"📊 Scan Complete! Detected {len(pdf_files)} PDF files")
    print("=" * 60)
    
    print("\n🚨 [HIGH PRIORITY FOR VLM CONVERSION] (Highly visual / scanned / fragmented text):")
    if not vlm_candidates:
        print("  ✅ No PDFs require VLM conversion.")
    else:
        # Sort by visual ratio descending (worst offenders first)
        vlm_candidates.sort(key=lambda x: x["visual_ratio"], reverse=True)
        for item in vlm_candidates:
            print(f"  📄 {item['name']}")
            print(f"     ➔ Total Pages: {item['pages']} | Avg Chars/Page: {item['avg_chars']} | Visual Page Ratio: {item['visual_ratio']:.1%}")

    print("\n✅ [HEALTHY TEXT PDFs] (Process directly with current Load_data.py):")
    if not healthy_pdfs:
        print("  None.")
    else:
        for name in healthy_pdfs:
            print(f"  🟢 {name}")

    if failed_pdfs:
        print("\n⚠️ [UNREADABLE / CORRUPTED FILES]:")
        for fail in failed_pdfs:
            print(f"  ❌ {fail}")
            
    print("\n💡 Tip: Only pass the PDFs listed under [HIGH PRIORITY FOR VLM CONVERSION] through the multimodal API script to generate Markdown.")

if __name__ == "__main__":
    # Pointing to your raw files directory
    raw_files_dir = "../knowledge_base/raw_files"
    
    if os.path.exists(raw_files_dir):
        # You can fine-tune these thresholds if necessary
        diagnose_pdfs(directory=raw_files_dir, text_threshold=300, visual_page_ratio=0.3)
    else:
        print(f"Please check if the folder path '{raw_files_dir}' is correct.")