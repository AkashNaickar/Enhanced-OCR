"""Extract text from a folder of images with Google Cloud Vision.

The heavy dependencies (``google-cloud-vision``, OpenCV, Pillow, tqdm) are
imported lazily inside the functions, so importing this module -- and running
the test suite -- works without a service account or the ML runtime installed.
"""

from __future__ import annotations

import argparse
import csv
import os

from enhanced_ocr.textutils import (
    clean_extracted_text,
    list_images,
    resolve_service_account_path,
)


def preprocess_image(image_path):
    """Deskew, threshold and sharpen an image, returning a temp file path.

    Falls back to *image_path* unchanged when the pre-processing dependencies
    are missing or processing fails.
    """
    try:
        import math
        import tempfile

        import cv2
        import numpy as np
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError as exc:
        print(f"Pre-processing dependencies unavailable ({exc}); using original image")
        return image_path

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read {image_path} with OpenCV, falling back to PIL")
            return image_path

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                if abs(angle) < 30:
                    angles.append(angle)

            if angles:
                median_angle = np.median(angles)

                if abs(median_angle) > 0.5:
                    (h, w) = gray.shape[:2]
                    center = (w // 2, h // 2)
                    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    gray = cv2.warpAffine(
                        gray,
                        matrix,
                        (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        temp_dir = tempfile.gettempdir()
        temp_filename = f"preprocessed_{os.path.basename(image_path)}"
        temp_path = os.path.join(temp_dir, temp_filename)

        cv2.imwrite(temp_path, binary)

        pil_img = Image.open(temp_path)

        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.2)

        pil_img = pil_img.filter(ImageFilter.SHARPEN)

        pil_img.save(temp_path)

        return temp_path

    except Exception as exc:  # noqa: BLE001 - keep the original per-image fallback
        print(f"Error preprocessing image {image_path}: {exc}")
        return image_path


def _build_client(credentials_path=None):
    """Create a Vision client, failing fast with a helpful message."""
    from google.cloud import vision

    path = credentials_path or resolve_service_account_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Service account file not found: {path}. Set GOOGLE_APPLICATION_CREDENTIALS "
            "in your environment (see .env.example) or pass --credentials."
        )
    return vision.ImageAnnotatorClient.from_service_account_file(path)


def _progress(iterable, description):
    """Wrap *iterable* in tqdm when available, else return it unchanged."""
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=description)


def _detect_text(client, image, image_context, image_file):
    """Run document detection with a text-detection fallback for one image."""
    try:
        doc_response = client.document_text_detection(image=image, image_context=image_context)

        if doc_response.error.message or not doc_response.text_annotations:
            response = client.text_detection(image=image, image_context=image_context)
        else:
            response = doc_response

        if response.error.message:
            print(f"Error with {image_file}: {response.error.message}")
            return ""
        if response.text_annotations:
            return clean_extracted_text(response.text_annotations[0].description)
        return ""
    except Exception as exc:  # noqa: BLE001 - one bad image must not abort the batch
        print(f"Error processing {image_file}: {exc}")
        return ""


def _write_results(results, output_file):
    """Write ``filename``/``text`` rows as TSV or CSV based on the extension."""
    delimiter = "\t" if output_file.lower().endswith(".tsv") else ","
    with open(output_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "text"], delimiter=delimiter)
        writer.writeheader()
        writer.writerows(results)


def extract_text_from_images(
    image_folder,
    output_file,
    use_preprocessing=True,
    client=None,
    credentials_path=None,
    vision_module=None,
):
    """OCR every supported image in *image_folder* and write the results.

    Args:
        image_folder: Directory containing the images.
        output_file: Destination path; ``.tsv`` selects tab separation, anything
            else uses commas.
        use_preprocessing: Run the OpenCV/Pillow deskew + threshold pipeline.
        client: Pre-built Vision client. Built from credentials when omitted.
        credentials_path: Service-account key path for the default client.
        vision_module: The ``google.cloud.vision`` module to build requests with.
            Resolved by import when omitted; injectable for tests.
    """
    if vision_module is None:
        from google.cloud import vision as vision_module

    if client is None:
        client = _build_client(credentials_path)

    image_files = list_images(image_folder)

    results = []
    temp_files = []

    for image_file in _progress(image_files, "Processing images"):
        image_path = os.path.join(image_folder, image_file)

        processed_image_path = image_path
        if use_preprocessing:
            processed_image_path = preprocess_image(image_path)
            if processed_image_path != image_path:
                temp_files.append(processed_image_path)

        with open(processed_image_path, "rb") as image_content:
            content = image_content.read()

        image = vision_module.Image(content=content)
        image_context = vision_module.ImageContext(language_hints=["en"])

        results.append(
            {
                "filename": image_file,
                "text": _detect_text(client, image, image_context, image_file),
            }
        )

    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except OSError:
            pass

    _write_results(results, output_file)

    print(f"Processed {len(results)} images. Results saved to {output_file}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract text from a folder of images using Google Cloud Vision."
    )
    parser.add_argument(
        "--image-folder",
        default="image_folder",
        help="Folder containing the images to OCR (default: image_folder).",
    )
    parser.add_argument(
        "--output",
        default="extracted_texts.tsv",
        help="Output path; .tsv is tab-separated, otherwise comma-separated.",
    )
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Skip the OpenCV/Pillow deskew and threshold step.",
    )
    parser.add_argument(
        "--credentials",
        default=None,
        help="Path to the service-account JSON key (default: env var or vision_ocr.json).",
    )
    args = parser.parse_args(argv)

    extract_text_from_images(
        image_folder=args.image_folder,
        output_file=args.output,
        use_preprocessing=not args.no_preprocessing,
        credentials_path=args.credentials,
    )


if __name__ == "__main__":
    main()
