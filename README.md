# Social Media Prediction Markets

A probability engine for social media and cultural milestone prediction markets.

**"Will MrBeast hit 300M subscribers?" "Will this app reach #1?" "Will the video hit 100M views?"**

Fits growth models to real data, runs Monte Carlo simulations, and detects edge in prediction market pricing — all with a beautiful dark-themed web dashboard.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **8 Data Collectors** — Social Blade, Google Trends, Reddit, Wikipedia pageviews, Wayback Machine, App Store, Twitch, Spotify
- **6 Growth Models** — Linear, exponential, logistic, viral, plateau, declining — auto-selected by AIC
- **Monte Carlo Simulation** — 10K-path stochastic simulation with viral shocks, plateau risk, and power law analysis
- **Survival Analysis** — Kaplan-Meier time-to-milestone from historical comparables
- **Edge Detection** — Mean reversion, cross-platform signals, controversy overcorrection, growth pattern mispricing
- **Web Dashboard** — Interactive dark-themed UI with real-time charts, probability gauges, and edge signal panels
- **CLI** — Full command-line interface for scripting and automation

---

## Installation on macOS

### Prerequisites

- **Python 3.10+** (check with `python3 --version`)
- **Homebrew** (recommended for managing dependencies)

### Step 1: Install Python (if needed)

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.10+
brew install python@3.12
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/powoso/Claude-social-media-viral-events.git
cd Claude-social-media-viral-events
```

### Step 3: Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install the Package

```bash
# Core package (CLI + probability engine)
pip install -e .

# With web dashboard
pip install -e ".[web]"

# With development tools (tests, linting)
pip install -e ".[dev,web]"
```

### Step 5: Configure API Keys (Optional)

Create a `.env` file or export environment variables for enhanced data collection:

```bash
# Social Blade API (optional — falls back to scraping)
export SOCIALBLADE_API_KEY="your_key_here"

# Reddit API (optional — enables Reddit collector)
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"

# Google Trends proxy (optional — helps avoid rate limits)
export GOOGLE_TRENDS_PROXY="http://your-proxy:port"
```

The system works fully without API keys using demo data and the web scraping fallbacks.

---

## Quick Start

### Launch the Web Dashboard

```bash
# Start the web server
smm-web

# Or run directly
python -m social_media_markets.web
```

Open **http://localhost:8000** in your browser.

The dashboard has 4 tabs:

| Tab | Description |
|-----|-------------|
| **Predict** | Run the full probability engine on a milestone question |
| **Growth Curves** | Fit and compare 6 growth models to any time series |
| **Simulate** | Monte Carlo simulation with confidence bands |
| **Edge Detector** | Find mispriced prediction market contracts |

### CLI Usage

```bash
# Collect data from Social Blade
smm collect MrBeast --platform youtube --source socialblade -o data.json

# Fit growth curves
smm fit data.json --all-models

# Predict probability of milestone
smm predict MrBeast 300000000 \
  --platform youtube \
  --metric subscribers \
  --deadline 2027-01-01 \
  --market-price 0.65

# Detect market edge
smm edge MrBeast 300000000 0.65 --platform youtube
```

---

## Architecture

```
social_media_markets/
├── collectors/          # Data ingestion from 8 sources
│   ├── socialblade.py   #   YouTube, TikTok, Instagram, Twitter
│   ├── google_trends.py #   Search interest (leading indicator)
│   ├── reddit_collector.py  # Subreddit growth & engagement
│   ├── wikipedia.py     #   Pageviews as attention proxy
│   ├── wayback.py       #   Historical snapshot reconstruction
│   ├── app_store.py     #   iTunes rankings & download estimates
│   ├── twitch.py        #   Streaming viewership
│   └── spotify.py       #   Chart positions & stream counts
├── features/            # Feature engineering
│   ├── growth_curves.py #   6-model fitting with AIC selection
│   ├── cross_platform.py    # Lagged correlation & Granger causality
│   ├── seasonality.py   #   FFT decomposition & monthly factors
│   ├── controversy.py   #   Spike detection & decay modeling
│   └── engagement.py    #   Cadence & viral hit rate analysis
├── probability/         # Probability engine
│   ├── engine.py        #   Main orchestrator (weighted ensemble)
│   ├── threshold.py     #   Growth extrapolation + uncertainty bands
│   ├── survival.py      #   Kaplan-Meier from comparables
│   └── simulation.py    #   Monte Carlo with viral shocks
├── edge/                # Edge detection
│   └── detector.py      #   5 bias detectors → buy/sell/pass
├── web/                 # Web dashboard
│   ├── __init__.py      #   FastAPI application & API routes
│   ├── templates/       #   Jinja2 HTML templates
│   └── static/          #   CSS & JavaScript
├── cli.py               # Command-line interface
├── config.py            # Configuration & enums
└── models.py            # Core data models
```

---

## How the Probability Engine Works

The engine combines three independent estimation methods:

1. **Growth Curve Extrapolation (25%)** — Fits the best parametric model, projects forward with expanding uncertainty bands, computes first-passage probability to target

2. **Survival Analysis (25%)** — Uses Kaplan-Meier on historical comparables (similar entities at similar growth stages) to estimate time-to-milestone distributions

3. **Monte Carlo Simulation (50%)** — Runs thousands of stochastic paths incorporating:
   - Parameter uncertainty (parametric bootstrap)
   - Power-law viral shocks (sudden jumps)
   - Plateau risk (structural growth ceiling)
   - Seasonal adjustments

The raw ensemble probability is then adjusted for:
- **Mean reversion** — exponential growth gets a 15% haircut
- **Cross-platform momentum** — TikTok acceleration boosts YouTube predictions
- **Controversy spikes** — fast-decaying spikes get discounted
- **Seasonality** — holiday tailwinds / summer headwinds

---

## Edge Detection Biases

The system detects 5 systematic biases in social media prediction markets:

| Bias | What Markets Do Wrong | Our Edge |
|------|----------------------|----------|
| **Recency** | Overweight last 2 weeks | Compare 14d vs 90d growth rate |
| **Platform Blindness** | Ignore cross-platform signals | Granger causality on Google Trends, TikTok → YouTube |
| **Controversy Panic** | Overreact to negative spikes | Model exponential decay with residual factor |
| **Exponential Extrapolation** | Assume growth continues forever | Detect logistic ceiling via AIC model selection |
| **Algorithm Ignorance** | Don't price platform changes | Structural break detection in growth patterns |

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=social_media_markets --cov-report=term-missing

# Just the web API tests
python -m pytest tests/test_web.py -v
```

---

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, NumPy, SciPy, pandas
- **Frontend**: Vanilla JS, Chart.js, CSS custom properties
- **Analysis**: scipy.optimize (curve fitting), scipy.signal (FFT), scipy.stats (survival)
- **Scraping**: httpx, BeautifulSoup4, lxml

---

## License

MIT
