"""
Code for main page of the app
"""

from io import BytesIO
from PIL import Image

import hydra
import streamlit as st

from src.app_helpers import get_ner_model, get_ner_processor
from src.invoice import process_invoice


@hydra.main(config_path="config/", config_name="config.yaml", version_base=None)
def app(config):

    # Initialize the session state
    if "processed_invoice" not in st.session_state:
        st.session_state.processed_invoice = None

    # Set the page config
    st.set_page_config(
        initial_sidebar_state="expanded",
        layout="wide",
    )

    # Set the title and header
    st.title("InvoiceXpert 🧾🧠")
    st.header("Your invoice data, unleashed")

    # Upload the invoice
    raw_invoice = st.sidebar.file_uploader(
        "Upload your invoice", type=["png", "jpg", "jpeg", "pdf"]
    )

    # Load the model and processor
    processor = get_ner_processor(config.models.retrieval.layoutlmv3)
    ner_model = get_ner_model(config.models.retrieval.layoutlmv3_invoice_ft)

    if raw_invoice:
        # Check if the invoice has been processed
        invoice_image = Image.open(BytesIO(raw_invoice.getvalue())).convert("RGB")
        if not st.session_state.processed_invoice:
            st.session_state.invoice_to_display = invoice_image
        # Display the invoice
        st.image(st.session_state.invoice_to_display, width=800)
        # Process the invoice
        do_process = st.button("Process invoice")
        if do_process:
            # Process the invoice
            processed_invoice, invoice_data = process_invoice(
                invoice_image, processor, ner_model
            )
            st.session_state.invoice_to_display = processed_invoice
            st.session_state.processed_invoice = True
            # Display extracted data
            st.markdown("### Invoice data")
            st.write(invoice_data)


if __name__ == "__main__":
    app()
