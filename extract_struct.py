import os
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO
from multiprocessing import Process, Queue, cpu_count
from huggingface_hub import hf_hub_download
import fitz  # PyMuPDF
from PIL import Image
from collections import defaultdict
import traceback
import logging
import io
import easyocr

# --- Configuration ---
INPUT_DIR = "input"
OUTPUT_DIR = "json"
MODEL_REPO = "hantian/yolo-doclaynet"
MODEL_FILE = "yolov12n-doclaynet.pt"
MODEL_CACHE = "/models"
NUM_WORKERS = min(cpu_count(), 5)
PADDING = 5

# Setup logging
logging.basicConfig(
    filename="processing.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(process)d %(message)s"
)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download YOLO model (once)
MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, cache_dir=MODEL_CACHE)

# --- Class Mapping ---
CLASS_MAP = {
    0: 'Caption', 1: 'Footnote', 2: 'Formula', 3: 'List-item', 4: 'Page-footer',
    5: 'Page-header', 6: 'Picture', 7: 'Section-header', 8: 'Table',
    9: 'Text', 10: 'Title'
}

# --- PDF Image Conversion ---
def convert_pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    return [Image.open(io.BytesIO(page.get_pixmap(dpi=100).tobytes("png"))) for page in doc]

# --- Worker Function ---
def worker(task_queue: Queue, model_path: str, result_queue: Queue):
    try:
        model = YOLO(model_path)
        ocr_model = easyocr.Reader(
            lang_list=['en'],
            model_storage_directory=MODEL_CACHE,
            gpu=False,
            verbose=False
        )
        result_queue.put("READY")
    except Exception as e:
        logging.error(f"Worker failed to load models: {e}")
        traceback.print_exc()
        result_queue.put("READY")
        return

    while True:
        item = task_queue.get()
        if item == "STOP":
            break
        try:
            pdf_name, page_idx, image_np, raw_page_bytes = item
            result = model(image_np)[0]
            detections = result.boxes.data.cpu().numpy()
            page = fitz.open("pdf", raw_page_bytes)[0]

            entries = []
            for det in detections:
                try:
                    x1, y1, x2, y2, conf, cls = det
                    class_id = int(cls)
                    class_name = CLASS_MAP.get(class_id, None)
                    if class_name not in {"Section-header", "Text", "List-item"}:
                        continue

                    x1, y1 = max(0, x1 - PADDING), max(0, y1 - PADDING)
                    x2, y2 = min(image_np.shape[1], x2 + PADDING), min(image_np.shape[0], y2 + PADDING)
                    if x1 >= x2 or y1 >= y2:
                        continue

                    # Convert image bbox to PDF coordinate system
                    img_h, img_w = image_np.shape[:2]
                    scale_x = page.rect.width / img_w
                    scale_y = page.rect.height / img_h
                    pdf_x1 = x1 * scale_x
                    pdf_y1 = y1 * scale_y
                    pdf_x2 = x2 * scale_x
                    pdf_y2 = y2 * scale_y

                    blocks = page.get_text("blocks")
                    collected_text = ""
                    for b in blocks:
                        bx1, by1, bx2, by2 = b[:4]
                        overlap = not (bx2 < pdf_x1 or bx1 > pdf_x2 or by2 < pdf_y1 or by1 > pdf_y2)
                        if overlap:
                            collected_text += b[4].strip() + " "

                    # Cleanup
                    text = collected_text.replace("\n", " ").replace("•", " ").replace("−", " ")
                    text = text.replace("*", " ").replace("·", " ").strip()
                    text = ' '.join(text.split())

                    if text:
                        entries.append({
                            "type": class_name,
                            "text": text,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "page": page_idx
                        })
                    else:
                        crop = image_np[int(y1):int(y2), int(x1):int(x2)]
                        ocr_result = ocr_model.readtext(crop, detail=1)
                        text = " ".join([res[1] for res in ocr_result]).strip()
                        conf = np.mean([res[2] for res in ocr_result]) if ocr_result else 0.5
                except Exception as e:
                    logging.error(f"Detection error in {pdf_name} page {page_idx}: {e}")
                    traceback.print_exc()

            result_queue.put((pdf_name, entries))
        except Exception as e:
            logging.error(f"Worker failure on {item}: {e}")
            traceback.print_exc()
            result_queue.put((item[0] if isinstance(item, tuple) else "unknown", []))

# --- Producer Function ---
def producer(pdf_path: str, task_queue: Queue):
    try:
        doc = fitz.open(pdf_path)
        base_name = os.path.basename(pdf_path)
        for idx, page in enumerate(doc):
            image = Image.open(io.BytesIO(page.get_pixmap(dpi=90).tobytes("png")))
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            single_page_pdf = fitz.open()
            single_page_pdf.insert_pdf(doc, from_page=idx, to_page=idx)
            page_bytes = single_page_pdf.write()
            task_queue.put((base_name, idx, image_np, page_bytes))

        return len(doc)
    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {e}")
        traceback.print_exc()
        return 0

# --- Structuring and Cleaning ---
def clean_and_structure_output(sorted_items):
    structured = []
    current_section = None
    section_headers_seen = set()

    for item in sorted_items:
        item_text = item["text"].strip()
        item_type = item["type"]

        if item_type in {"Title", "Section-header"}:
            current_section = {
                "section-header": item_text,
                "page": item["page"],
                "content": ""
            }
            structured.append(current_section)
            section_headers_seen.add(item_text)

        elif current_section:
            # Discard if content matches any section header exactly
            if item_text in section_headers_seen:
                continue
            current_section["content"] += item_text + " "

    # Final cleanup of whitespace
    for section in structured:
        section["content"] = ' '.join(section["content"].split())

    return structured

def extractor(documents):
    if not documents:
        logging.error("No PDF files found.")
        exit(1)
    try:
        logging.info("Starting processing...")

        task_queue = Queue()
        result_queue = Queue()
        workers = []

        for i in range(NUM_WORKERS):
            p = Process(target=worker, args=(task_queue, MODEL_PATH, result_queue))
            p.start()
            workers.append(p)

        for _ in range(NUM_WORKERS):
            result_queue.get()
        logging.info("All workers are ready.")

        start_time = time.time()

        

        total_pages = 0
        pdf_page_counts = {}
        for file in documents:
            full_path = os.path.join(INPUT_DIR, file)
            pages = producer(full_path, task_queue)
            pdf_page_counts[file] = pages
            total_pages += pages

        logging.info(f"Total pages queued: {total_pages}")

        results_by_pdf = defaultdict(list)
        received_pages = 0
        received_pages_per_pdf = defaultdict(int)

        while received_pages < total_pages:
            try:
                pdf_name, data = result_queue.get(timeout=60)
                results_by_pdf[pdf_name].extend(data)
                received_pages_per_pdf[pdf_name] += 1
                received_pages += 1

                if received_pages_per_pdf[pdf_name] == pdf_page_counts[pdf_name]:
                    try:
                        base_name = os.path.splitext(os.path.basename(pdf_name))[0]
                        out_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")

                        sorted_items = sorted(
                            results_by_pdf[pdf_name],
                            key=lambda x: (x["page"], x["bbox"][1])
                        )

                        cleaned = clean_and_structure_output(sorted_items)

                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(cleaned, f, indent=2, ensure_ascii=False)

                        logging.info(f"Saved cleaned JSON for {base_name} to {out_path}")
                        del results_by_pdf[pdf_name]
                        del received_pages_per_pdf[pdf_name]
                    except Exception as e:
                        logging.error(f"Error saving JSON for {pdf_name}: {e}")
                        traceback.print_exc()
            except Exception as e:
                logging.error("Timeout or error waiting for results.")
                traceback.print_exc()
                break

        for _ in range(NUM_WORKERS):
            task_queue.put("STOP")
        for p in workers:
            p.join()

        logging.info(f"All PDFs processed in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logging.critical(f"Fatal error in main: {e}")
        traceback.print_exc()
