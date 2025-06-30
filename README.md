# HealthAI - Intelligent Healthcare Assistant 🏥🤖

## 🌟 Overview

**HealthAI** is an intelligent healthcare assistant that combines **IBM Watson Machine Learning** and **Generative AI** to deliver accurate, personalized medical support. Built with Streamlit and powered by IBM's Granite-13b-instruct-v2 model, HealthAI offers users the ability to chat with a virtual assistant for health questions, get symptom-based disease predictions, receive customized treatment suggestions, and track their health through visual analytics.

### 🚀 Live Demo
Experience HealthAI: https://healthaiassistant.streamlit.app/
(https://healthaiassistant.streamlit.app/)

## ✨ Key Features

### 1. 💬 Chat with AI Doctor
- Interactive medical consultation interface
- Real-time AI-powered responses to health queries
- Context-aware answers using IBM's Granite model
- Basic medical information and guidance

### 2. 🔍 Disease Predictor
- Symptom-based disease prediction
- Input symptoms (e.g., "fever and cough")
- AI predicts likely conditions (Flu, COVID-19, etc.)
- Rule-based matching with potential for ML enhancement

### 3. 📋 Treatment Plan Generator
- Generate treatment plans for common diseases
- Input disease name to receive treatment outline
- Includes home remedies and dietary suggestions
- General-purpose medical guidance

### 4. 📊 Health Analytics Dashboard
- Visual health metrics tracking
- Interactive charts and visualizations
- Time series health data analysis
- Built with Plotly for dynamic displays

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Model**: IBM Granite-13b-instruct-v2
- **API**: IBM Watson Machine Learning
- **Visualizations**: Plotly
- **Backend**: Python
- **Deployment**: Streamlit Cloud

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  Python Backend │────│  IBM Granite    │
│   (Frontend)    │    │                 │    │   AI Model      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                    ┌─────────────────┐
                    │   Health Data   │
                    │   Visualizer    │
                    └─────────────────┘
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- IBM Watson Machine Learning API credentials
- Streamlit

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/healthai.git
cd healthai
```

2. **Install dependencies**
```bash
pip install streamlit
pip install plotly
pip install ibm-watson-machine-learning
# Add other required packages
```

3. **Set up environment variables**
```bash
# Create a .env file and add your IBM API credentials
IBM_API_KEY=your_api_key_here
```

4. **Run the application**
```bash
streamlit run app.py
```

## 🎯 Usage Scenarios

### Scenario 1: Medical Consultation
```
User Input: "I have been experiencing headaches and fatigue"
AI Response: Provides potential causes and recommendations
```

### Scenario 2: Disease Prediction
```
User Input: "fever and cough"
AI Prediction: "Possible conditions: Flu, Common Cold, COVID-19"
```

### Scenario 3: Treatment Planning
```
User Input: "Malaria"
AI Output: Comprehensive treatment outline with care instructions
```

## 📸 Screenshots

### Chat Interface
![Chat with AI Doctor](path/to/chat-screenshot.jpg)

### Disease Prediction
![Disease Predictor](path/to/prediction-screenshot.jpg)

### Treatment Generator
![Treatment Generator](path/to/treatment-screenshot.jpg)

### Health Analytics
![Health Analytics](path/to/analytics-screenshot.jpg)

## 🔧 Development Workflow

### Phase 1: Setup & Architecture
- ✅ Model selection (IBM Granite-13b-instruct-v2)
- ✅ Architecture design
- ✅ Development environment setup

### Phase 2: Core Development
- ✅ Patient chat functionality
- ✅ Disease prediction logic
- ✅ Treatment plan generation
- ✅ Health analytics implementation

### Phase 3: Frontend & Integration
- ✅ Streamlit UI development
- ✅ AI model integration
- ✅ Dynamic visualizations

### Phase 4: Deployment
- ✅ Environment configuration
- ✅ Streamlit Cloud deployment
- ✅ Testing and documentation

## ⚠️ Important Disclaimers

- **Not a Medical Device**: All responses are AI-generated and may lack medical accuracy
- **Not a Replacement**: This tool does not replace professional medical consultation
- **Educational Purpose**: Designed for informational and educational use only
- **Seek Professional Care**: Always consult healthcare professionals for medical concerns

## 🔮 Future Enhancements

- [ ] **Enhanced ML Models**: Integrate scikit-learn classifiers with real medical datasets
- [ ] **Advanced Analytics**: Comprehensive health dashboard with predictive analytics
- [ ] **User History**: Session-based interaction history and personalization
- [ ] **Real Dataset Integration**: SymCat or WHO dataset integration
- [ ] **Mobile Optimization**: Responsive design for mobile devices
- [ ] **Multi-language Support**: International accessibility

## 🚧 Current Limitations

- Requires internet connectivity for IBM API access
- API rate limits may apply
- Responses are general-purpose, not personalized medical advice
- Limited to common conditions and symptoms

## 👥 Contributors

- **G. Venkat Reddy** - Lead Developer
- **Guvvala Dhanush Kumar** - Backend Development
- **Koka Yogesh** - Frontend & UI
- **Vasavi T** - Testing & Documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

If you have any questions or need support, please open an issue on GitHub or contact the development team.

## 🙏 Acknowledgments

- IBM Watson for providing the Granite AI model
- Streamlit for the excellent web framework
- Plotly for powerful visualization capabilities
- The open-source community for inspiration and resources
