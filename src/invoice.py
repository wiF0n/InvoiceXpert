from typing import List, Tuple, Any, Dict

import numpy as np

from PIL import Image, ImageDraw, ImageFont

"""
Functionality for visualizing NER results on invoices.
"""


_LABEL2COLOR = {
    "B-ABN": "blue",
    "B-BILLER": "blue",
    "B-BILLER_ADDRESS": "green",
    "B-BILLER_POST_CODE": "orange",
    "B-DUE_DATE": "blue",
    "B-GST": "green",
    "B-INVOICE_DATE": "violet",
    "B-INVOICE_NUMBER": "orange",
    "B-SUBTOTAL": "green",
    "B-TOTAL": "blue",
    "I-BILLER_ADDRESS": "blue",
    "O": "orange",
}


_ITEMS_OF_INTEREST = ["B-INVOICE_DATE", "B-INVOICE_NUMBER", "B-TOTAL"]


def _unnormalize_box(
    bbox: Tuple[float, float, float, float], width: float, height: float
) -> List[float]:
    """
    Unnormalize the bounding box coordinates based on the image width and height.

    Args:
        bbox (Tuple[float, float, float, float]): The normalized bounding box coordinates (x_min, y_min, x_max, y_max).
        width (float): The width of the image.
        height (float): The height of the image.

    Returns:
        List[float]: The unnormalized bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def _parse_label(label: str) -> str:
    """
    Parse the label to remove the "B-" prefix.
    """
    label = label[2:]
    if not label:
        return "other"
    return label


def process_invoice(
    image: Image.Image,
    processor: Any,
    model: Any,
    items_of_interest: List[str] = _ITEMS_OF_INTEREST,
) -> Tuple[Image.Image, List[Dict[str, str]]]:
    """
    Process the image by encoding, making predictions, and drawing bounding boxes and labels.

    Args:
        image: The input image.
        processor: The image processor (LayoutLMv3 processor).
        model: The model for making predictions (LayoutLMv3 model).
        id2label: A dictionary mapping label IDs to labels.
        label2color: A dictionary mapping labels to colors.
        items_of_interest: A list of labels to consider.

    Returns:
        Image.Image: The processed image.
        List[Dict[str, str]]: The final output.
    """
    # get image dimensions
    width, height = image.size

    # encode
    encoding = processor(
        image, truncation=True, return_offsets_mapping=True, return_tensors="pt"
    )
    offset_mapping = encoding.pop("offset_mapping").squeeze().tolist()

    # forward pass
    outputs = model(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()
    input_ids = encoding.input_ids.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping)[:, 0] != 0
    true_predictions = [
        model.config.id2label[pred]
        for idx, pred in enumerate(predictions)
        if not is_subword[idx]
    ]
    true_boxes = [
        _unnormalize_box(box, width, height)
        for idx, box in enumerate(token_boxes)
        if not is_subword[idx]
    ]

    # Merge subwords into words
    word_start_idx = []
    for idx, sub_word in enumerate(offset_mapping):
        if sub_word[0] == 0:
            word_start_idx.append(idx)

    words = []
    for idx, word_idx in enumerate(word_start_idx):
        if idx + 1 == len(word_start_idx):
            break
        else:
            end_idx = idx + 1
        word_input_ids = input_ids[word_idx : word_start_idx[end_idx]]
        words.append(processor.tokenizer.decode(word_input_ids))

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Instantiate the final output
    final_output = []

    for word, predicted_label, box in zip(words, true_predictions, true_boxes):
        if predicted_label in items_of_interest:
            # parse the label
            pretty_predicted_label = _parse_label(predicted_label)

            # Construct the final output
            final_prediction = {
                "word": word,
                "label": pretty_predicted_label,
                "box": box,
            }
            final_output.append(final_prediction)

            # Construct the image
            color = _LABEL2COLOR.get(predicted_label)
            draw.rectangle(box, outline=color)
            draw.text(
                (box[0] + 10, box[1] - 10),
                text=pretty_predicted_label,
                fill=color,
                font=font,
            )

    return image, final_output
