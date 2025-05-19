# Patient Record Summarizer

A powerful Streamlit application that processes and summarizes patient medical records using advanced AI models. This application helps healthcare professionals quickly extract and understand key information from patient PDF reports.

## 🚀 Features

- **PDF Processing**: Extract text from patient medical reports in PDF format
- **Medical Entity Recognition**: Identify and highlight medical terms, conditions, and medications
- **AI-Powered Summarization**:
  - Choose between DeepSeek R1 or GPT-4 for summarization
  - Structured medical summaries
  - Entity-aware processing
- **Interactive Interface**:
  - Upload multiple PDFs
  - Combine summaries from multiple reports
  - View extracted entities and model reasoning
- **Real-time Processing**: Immediate feedback and results

## 🛠️ Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/DhyanVGowda/patient-record-summarizer.git
   cd patient-record-summarizer
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root with your API keys:

   ```
   REPLICATE_API_TOKEN=your_replicate_api_token
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_patient_summary_app.py
   ```

## 📋 Requirements

- Python 3.7+
- Streamlit
- OpenAI API key
- Replicate API key
- Other dependencies listed in `requirements.txt`

## 🔧 Usage

1. Launch the application
2. Upload one or more patient PDF reports
3. Select the summarization model (DeepSeek R1 or GPT-4)
4. Choose whether to combine multiple reports
5. Click "Start Summarization"
6. View the extracted text, medical entities, and AI-generated summary

## 🏗️ Architecture

The application is built with a clean separation of concerns:

- `streamlit_patient_summary_app.py`: User interface and interaction
- `patient_summary_logic.py`: Core processing logic and AI integration

## 🔒 Security

- API keys are stored securely in environment variables
- No patient data is stored permanently
- All processing is done in memory

## 🌐 Deployment

This application is deployed on Streamlit Cloud. Visit [app link] to use it online.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. Always verify AI-generated summaries with qualified medical professionals.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
