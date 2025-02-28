# women_maintainance
# Women's Health Personalization Platform 🚀

This Streamlit-based web application provides **personalized health recommendations** for women based on their health parameters using **AI-powered clustering** and **Gemini AI**.

---

## 🌟 Features
- 📊 **Clustering Algorithms:** Uses **Hierarchical Clustering** and **Gaussian Mixture Model (GMM)** for health segmentation.
- 🤖 **AI-Powered Recommendations:** Leverages **Google Gemini AI** to generate health advice based on user inputs.
- 🎛 **Interactive UI:** Streamlit-powered interface with user-friendly sliders and text inputs.

---

## 🛠 Installation

1️⃣ **Clone the Repository**
```
git clone https://github.com/yourusername/womens-health-ai.git
cd womens-health-ai
```

2️⃣ Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3️⃣ Install Dependencies

```bash
pip install -r dependencies.txt
```
4️⃣ Set Up Environment Variables
Create a .env file in the project folder and add:
```
API_KEY=your_gemini_api_key_here
```
5️⃣ Run the Application
```
streamlit run app.py
```
📂 File Structure
```
├── app.py                   # Main application script
├── women_health_data.csv     # Dataset (Make sure it's available)
├── dependencies.txt          # Required dependencies
├── .env                      # API Key (Not included in GitHub)
└── README.md                 # Project documentation
```
🔥 Usage
Enter your health parameters using the sliders.
Describe any health concerns in the text box.
Submit your data to receive AI-generated personalized health advice.
View your segmentation results based on clustering.
🛠 Technologies Used
Python 🐍
Streamlit 🎛
Google Gemini AI 🤖
Scikit-learn & SciPy 📊
Pandas & NumPy 🏗
📌 Notes
Google Gemini API Key is required for AI-generated recommendations.
Ensure women_health_data.csv is available in the project directory.
📜 License
This project is open-source under the MIT License.

🤝 Contributions
Contributions are welcome! Feel free to fork the repo, create an issue, or submit a PR.

📧 Contact
For any questions or suggestions, reach out via GitHub Issues.
