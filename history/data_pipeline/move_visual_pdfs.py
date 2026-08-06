import os
import shutil
from pathlib import Path

# Set directory paths
base_dir = "../knowledge_base"
source_dir = os.path.join(base_dir, "raw_files")
target_dir = os.path.join(base_dir, "vlm_visual_pdfs")

# These are the 50 PDF filenames we identified that require VLM processing
files_to_move = {
    "The GEMS State Model.pdf",
    "LGBTQ+ Advance Care Planning.pdf",
    "LGBTQ+ Caregiver Rights.pdf",
    "Communication _ Alzheimer_s Association.pdf",
    "Do_s and Don_ts of Communication and Dementia - Alzheimer_s San Diego.pdf",
    "Medicare _ Alzheimer_s Association.pdf",
    "Early _ Younger-Onset Alzheimer_s _ Alzheimer_s Association.pdf",
    "30 Activities for Older Adults with Dementia _ TheKey - TheKey.pdf",
    "6-hours-Dementia-Care-Challenges.pdf",
    "Mild Cognitive Impairment - Module - Region 9 AAA.pdf",
    "Noticing Problems With Toileting - Module - Region 9 AAA.pdf",
    "Sleeping Medications_ Know The Risks - Module - Region 9 AAA.pdf",
    "Bathing & Showering_ The Basics - Module - Region 9 AAA.pdf",
    "4 Skills For Personal Care Tasks - Module - Region 9 AAA.pdf",
    "Blood Thinners_ Do_s & Don_ts - Module - Region 9 AAA.pdf",
    "After A Fall_ 3 Things To Do - Module - Region 9 AAA.pdf",
    "Toileting_ The Basics - Module - Region 9 AAA.pdf",
    "Are You Uncomfortable Helping With Toile - Module - Region 9 AAA.pdf",
    "Paying For Medications - Module - Region 9 AAA.pdf",
    "5 Skills For Mobility & Fall Prevention - Module - Region 9 AAA.pdf",
    "Hands-on Help With Oral Care - Module - Region 9 AAA.pdf",
    "Skin Care - Module - Region 9 AAA.pdf",
    "3 Ways To Encourage A Shower - Module - Region 9 AAA.pdf",
    "4 Skills For Managing Medications - Module - Region 9 AAA.pdf",
    "Poor Oral Care - Module - Region 9 AAA.pdf",
    "3 Skills For Discharge & Transitions - Module - Region 9 AAA.pdf",
    "Plan De Bienestar Para Cuidadores - Module - Region 9 AAA.pdf",
    "Sit To Stand Transfers - Module - Region 9 AAA.pdf",
    "Transition To Nursing Home - Module - Region 9 AAA.pdf",
    "Speaking Up For Your Care Recipient - Module - Region 9 AAA.pdf",
    "Challenging your brain.pdf",
    "Dementia-Care-Partner-Resources-Manual.pdf",
    "Spiritual_Care_of_the_Person_with_Dementia_Hicks_no_videos_Jan2017.pdf",
    "Staying Healthy_ Risks for Family Caregivers for Dementia Patients.pdf",
    "Anxiety and dementia.pdf",
    "Reducing Social Isolation with Technology.pdf",
    "Medications That Increase Fall Risk - Module - Region 9 AAA.pdf",
    "What Are ADLs_ - Module - Region 9 AAA.pdf",
    "Easy Exercises To Do At Home - Module - Region 9 AAA.pdf",
    "Vascular Dementia - Module - Region 9 AAA.pdf",
    "Medications_ Dementia Safety - Module - Region 9 AAA.pdf",
    "Outdoor Activities With Your Care Recipi - Module - Region 9 AAA.pdf",
    "Tips For A Safer Home - Module - Region 9 AAA.pdf",
    "Cognitive Changes After Surgery - Module - Region 9 AAA.pdf",
    "Organize Medications_ 4 Easy Tips - Module - Region 9 AAA.pdf",
    "Changing Your Care Recipient In Bed - Module - Region 9 AAA.pdf",
    "Dementia VA.pdf",
    "Managing finances for people with dementia.pdf",
    "How to reduce the stress when caring for someone with dementia_ Newsroom - UT Southwestern, Dallas, Texas.pdf",
    "Suspicions and Delusions.pdf"
}

def move_files_recursively():
    # Create the target directory automatically if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"📁 Created new folder: {target_dir}")

    success_count = 0
    found_files = set()

    print(f"🚚 Starting deep recursive search in {source_dir}...\n")

    # Use rglob("*") to recursively search all files in all subdirectories
    for filepath in Path(source_dir).rglob("*"):
        if filepath.is_file() and filepath.name in files_to_move:
            dst_path = os.path.join(target_dir, filepath.name)

            try:
                # shutil.move will cut and paste the file
                shutil.move(str(filepath), dst_path)
                print(f"  ✅ Successfully moved: {filepath.name}")
                print(f"     (Found in: {filepath.parent})")
                
                success_count += 1
                found_files.add(filepath.name)
            except Exception as e:
                print(f"  ❌ Failed to move {filepath.name}: {e}")

    # Calculate if any files were missed
    missing_files = files_to_move - found_files

    print("\n" + "="*60)
    print(f"🎉 Operation complete! Successfully extracted {success_count} files to {target_dir}")
    print("="*60)

    if missing_files:
        print("\n⚠️ The following files were still not found anywhere in the directory tree:")
        for f in missing_files:
            print(f"  - {f}")

if __name__ == "__main__":
    if not os.path.exists(source_dir):
        print(f"❌ Source folder not found: {source_dir}. Please check your current working directory.")
    else:
        move_files_recursively()