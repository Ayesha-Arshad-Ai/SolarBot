# ☀️ SolarBot
### AI-Powered Solar System Recommendation Engine

SolarBot is an AI-assisted backend system that generates solar energy system recommendations based on **user location, appliance usage, and solar radiation data**.

The system integrates **FastAPI, geolocation services, solar radiation APIs, and an LLM-based estimation model** to provide intelligent solar sizing recommendations.

This project demonstrates how **AI inference and engineering calculations can be combined to build practical renewable-energy tools.**

---

# 🚀 Overview

SolarBot calculates an optimal solar energy setup by combining:

- geographic location
- energy consumption estimates
- solar radiation data

The API processes user inputs and returns recommendations such as:

- solar system capacity (kW)
- number of panels required
- battery storage capacity
- estimated daily energy generation

This backend service can be integrated into:

- solar installer platforms
- renewable energy dashboards
- solar calculator web apps
- smart home energy tools

---

# ✨ Features

- AI-assisted appliance power estimation
- Location-based solar radiation analysis
- Solar system sizing calculations
- Panel count estimation
- Battery storage recommendation
- REST API architecture
- Automatic API documentation

---

# 🧠 AI Pipeline

SolarBot combines **AI predictions with deterministic energy calculations**.

### 1. Input Processing
User provides location and appliance usage.

### 2. Geolocation
Location names are converted into geographic coordinates using **Geopy**.

### 3. Solar Radiation Data
Solar irradiance information is retrieved from the **Open-Meteo API**.

### 4. AI Consumption Estimation
If appliance wattage is unknown, an **LLM (Groq API)** estimates power consumption.

### 5. Solar System Calculation
The system calculates:

- daily electricity demand
- solar capacity requirements
- number of solar panels
- battery storage
- expected system output

---

# 🧰 Tech Stack

| Category | Technology |
|--------|-------------|
| Backend Framework | FastAPI |
| Programming Language | Python |
| AI Integration | Groq LLM API |
| Solar Data | Open-Meteo API |
| Geolocation | Geopy |
| Data Validation | Pydantic |
| Environment Config | python-dotenv |

---

# 🏗 System Architecture

```
User Input
(Location + Appliances)
        │
        ▼
FastAPI Backend
        │
        ├── Geopy → Location Coordinates
        │
        ├── Open-Meteo API → Solar Radiation Data
        │
        ├── Groq LLM → Appliance Wattage Estimation
        │
        ▼
Solar Calculation Engine
        │
        ▼
Solar System Recommendation
```

---

# 📂 Project Structure

```
SolarBot
│
├── solarbot.py
│   FastAPI API endpoints
│
├── solar_function.py
│   Solar radiation calculations
│   Geolocation utilities
│
├── requirements.txt
│   Python dependencies
│
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```
git clone https://github.com/yourusername/solarbot.git
cd solarbot
```

Install dependencies

```
pip install -r requirements.txt
```

Create `.env`

```
GROQ_API_KEY=your_api_key_here
```

Run the API server

```
uvicorn solarbot:app --reload
```

Server will start at

```
http://127.0.0.1:8000
```

---

# 📖 API Documentation

FastAPI automatically generates interactive API documentation.

Open in browser:

```
http://localhost:8000/docs
```



# 💡 Skills Demonstrated

This project showcases:

- AI-assisted backend systems
- LLM integration in APIs
- FastAPI microservice architecture
- external API integration
- geolocation services
- renewable energy modeling
- production-ready API design


