import os
import re
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path as cfp


def convert_gray_to_white(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)
    inverted_binary = cv2.bitwise_not(binary)
    white_background = cv2.cvtColor(inverted_binary, cv2.COLOR_GRAY2BGR)
    return white_background


def process_box(cv_image, box):
    box = box.split(" ")
    height, width, _ = cv_image.shape
    x1, y1, x2, y2 = int(box[1]), height - int(box[2]), int(box[3]), height - int(box[4])
    letter_image = cv_image[y2:y1, x1:x2]
    letter = re.sub(r'[\\/\[\].:*?"<>|#^%\'+\-1234567890]', '_', box[0])
    return letter, letter_image


def extract_characters(cv_image, image, page_num):
    boxes = pytesseract.image_to_boxes(image, lang='heb', config=r"--psm 6 --oem 1")
    for index, box in enumerate(boxes.splitlines()):
        letter, letter_image = process_box(cv_image, box)
        yield letter, letter_image, page_num, index


def save_characters(output_dir, characters):
    for letter, letter_image, page_num, index in characters:
        letter_dir = os.path.join(output_dir, str(ord(letter)))
        os.makedirs(letter_dir, exist_ok=True)
        cv2.imwrite(os.path.join(letter_dir, f"page_{page_num}_index_{index}.png"), letter_image)


def extract_letters(image_path):
    output_dir = 'letter_images'
    os.makedirs(output_dir, exist_ok=True)
    pages = cfp(image_path)
    print(f"Total pages: {len(pages)}")
    for page_num, page in enumerate(pages):
        if page_num <= 10:
            process_page(output_dir, page, page_num)


def process_page(output_dir, page, page_num):
    cv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    img = convert_gray_to_white(cv_image)
    characters = extract_characters(cv_image, img, page_num)
    save_characters(output_dir, characters)
    print(f"Page {page_num + 1}: Characters extracted and saved")


if __name__ == "__main__":
    pdf_path = 'torah1.pdf'
    extract_letters(pdf_path)
