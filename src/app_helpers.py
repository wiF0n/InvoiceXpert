"""
Helper functionality for Streamlit app.
"""

import streamlit as st
from transformers import LayoutLMv3ForTokenClassification, AutoProcessor

from src.config import init_config as _init_config


@st.cache_resource
def get_ner_model(name: str) -> LayoutLMv3ForTokenClassification:
    """
    Load the NER model from the Hugging Face model hub.
    """
    return LayoutLMv3ForTokenClassification.from_pretrained(name)


@st.cache_resource
def get_ner_processor(name: str) -> AutoProcessor:
    """
    Load the NER processor from the Hugging Face model hub.
    """
    return AutoProcessor.from_pretrained(name)
