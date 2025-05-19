import pdfplumber
import torch
import replicate
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os
import spacy
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class PatientSummaryProcessor:
    def __init__(self, replicate_api_token=None, openai_api_key=None):
        self.replicate_api_token = replicate_api_token or os.getenv('REPLICATE_API_TOKEN')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.replicate_api_token:
            raise ValueError("REPLICATE_API_TOKEN not found. Please set it in .env file or pass it to the constructor.")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found. Please set it in .env file or pass it to the constructor.")
            
        os.environ["REPLICATE_API_TOKEN"] = self.replicate_api_token
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        self.CLINICAL_BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(self.CLINICAL_BERT_MODEL)
        self.model = AutoModel.from_pretrained(self.CLINICAL_BERT_MODEL)
        self.model.eval()
        
        self.nlp = spacy.load("en_ner_bc5cdr_md")

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {e}")
        return text

    def chunk_text(self, text, max_tokens=512):
        words = text.split()
        return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def sort_chunks_by_relevance(self, chunks, anchor="patient condition summary"):
        anchor_embedding = self.get_embedding(anchor)
        embeddings = [self.get_embedding(chunk) for chunk in chunks]
        scores = cosine_similarity([anchor_embedding], embeddings)[0]
        return [x for _, x in sorted(zip(scores, chunks), key=lambda p: p[0], reverse=True)][:5]

    def extract_medical_entities(self, text):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def clean_summary_output(self, text):
        return re.sub(
            r"(?i)(okay,?\s+let'?s\s+(tackle|analyze|look at|parse)[\s\S]*?)(Structured Medical Summary|Patient Information|Patient Overview)",
            r"\3",
            text,
            count=1
        ).strip()

    def summarize_with_deepseek(self, chunks, entities):
        entity_context = "\n".join([f"- {text} ({label})" for text, label in entities])
        summaries, thoughts = [], []
        for chunk in chunks:
            prompt = f"Known medical entities:\n{entity_context}\n\nSummarize the following clinical patient record in a structured medical format:\n{chunk}"
            output = replicate.run("deepseek-ai/deepseek-r1", input={"prompt": prompt, "max_new_tokens": 400})
            full = "".join(output)
            summaries.append(self.clean_summary_output(full))
            thoughts.append(f"---\n**Prompt:**\n{prompt}\n\n**LLM Output:**\n{full}")
        return "\n\n".join(summaries), "\n\n".join(thoughts)

    def summarize_with_gpt4o(self, chunks, entities):
        entity_context = "\n".join([f"- {text} ({label})" for text, label in entities])
        summaries, thoughts = [], []
        for chunk in chunks:
            prompt = f"Known medical entities:\n{entity_context}\n\nSummarize the following clinical patient record in a structured medical format:\n{chunk}"
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a clinical expert creating structured medical summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400
            )
            full = response.choices[0].message.content.strip()
            summaries.append(self.clean_summary_output(full))
            thoughts.append(f"---\n**Prompt:**\n{prompt}\n\n**LLM Output:**\n{full}")
        return "\n\n".join(summaries), "\n\n".join(thoughts)

    def highlight_entities(self, text):
        doc = self.nlp(text)
        for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
            text = text[:ent.start_char] + f"<mark title='{ent.label_}'>" + text[ent.start_char:ent.end_char] + "</mark>" + text[ent.end_char:]
        return text

    def process_pdf(self, pdf_file, model_choice, combine=False):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_file.seek(0)
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        
        try:
            text = self.extract_text_from_pdf(tmp_path)
            entities = self.extract_medical_entities(text)
            chunks = self.chunk_text(text)
            sorted_chunks = self.sort_chunks_by_relevance(chunks)
            
            if model_choice == "DeepSeek R1":
                summary, debug = self.summarize_with_deepseek(sorted_chunks, entities)
            else:
                summary, debug = self.summarize_with_gpt4o(sorted_chunks, entities)
            
            return {
                'text': text,
                'entities': entities,
                'summary': summary,
                'debug': debug,
                'highlighted_summary': self.highlight_entities(summary)
            }
        finally:
            os.remove(tmp_path) 