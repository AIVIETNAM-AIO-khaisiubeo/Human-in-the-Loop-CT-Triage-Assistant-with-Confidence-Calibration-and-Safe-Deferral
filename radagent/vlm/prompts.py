"""Prompts for the vision-language second reader (versioned: a change invalidates cached reads)."""
from __future__ import annotations

PROMPT_VERSION = "v1"

INSTRUCTION = ("You are a radiologist reviewing an abdominal contrast-enhanced CT scan. You are analyzing "
               "{n} axial CT slices that cover the liver, ordered from superior to inferior. Please review "
               "the slices provided below carefully.")

SCORE_QUERY = ("\n\nBased on the visual evidence in the slices provided above, is there at least one focal "
               "liver lesion suspicious for a liver tumour? Answer with a single word: yes or no.")

DESCRIBE_QUERY = ("\n\nBased on the visual evidence in the slices provided above, describe the liver. Comment "
                  "on hypodense or hyperdense focal lesions and other hepatic irregularities, their approximate "
                  "number and location. Do not comment on findings outside the liver and do not estimate sizes. "
                  "Be concise (at most five sentences) and conclude with 'Final Answer: yes' if a liver tumour "
                  "is suspected or 'Final Answer: no' if not.")


def build_content(n_images: int, query: str, image_items: list) -> list[dict]:
    """Chat content in the layout of Google's CT notebook: instruction, (image, 'SLICE i')*, query."""
    content = [{"type": "text", "text": INSTRUCTION.format(n=n_images)}]
    for i, item in enumerate(image_items, 1):
        content.append(item)
        content.append({"type": "text", "text": f"SLICE {i}"})
    content.append({"type": "text", "text": query})
    return content
