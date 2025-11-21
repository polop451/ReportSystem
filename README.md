# ReportSystem

## Overview

**ReportSystem** is an API service for detecting bad words, spam, and gambling-related content in both Thai and English texts. It also supports custom NLP model training for more advanced and adaptive content moderation.

## Features

- Detects bad words in Thai and English using customizable word lists.
- Detects spam and gambling-related keywords.
- Supports NLP model prediction for text classification (e.g., spam/gambling detection).
- Allows users to train and upload their own NLP models using labeled datasets.
- RESTful API built with FastAPI.

## Technology Stack

- **Python 3**
- **FastAPI** (web framework)
- **scikit-learn** (NLP model training)
- **pandas** (data handling)
- **joblib** (model serialization)
- **Uvicorn** (ASGI server)

## How to Use

### 1. Install dependencies

```sh
pip install -r [requirements.txt](http://_vscodecontentref_/0)
