# Excel Data Assistant

A conversational interface platform for analyzing Excel files using natural language queries.

## Features

- **Excel Upload**: Handles any Excel format with automatic data cleaning
- **Natural Language Queries**: Ask questions about your data in plain English
- **Data Visualization**: Automatic charts and tables for query results
- **Robust Data Processing**: Handles unnamed columns, dirty data, and inconsistent formatting

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Google Gemini API key

### Setup

1. **Backend Setup**:
   ```bash
   cd backend
   # Add your Gemini API key to .env file
   # Edit .env and replace 'your_gemini_api_key_here' with your actual key
   start.bat  # Windows
   ```

2. **Frontend Setup** (in new terminal):
   ```bash
   cd frontend
   start.bat  # Windows
   ```

3. **Access the app**: Open http://localhost:5173

## Usage

1. Upload any Excel file (.xlsx or .xls)
2. Ask questions like:
   - "What is the average sales by region?"
   - "Show me the top 10 customers"
   - "How many orders were placed last month?"
3. View results in tables and charts

## Architecture

- **Frontend**: React + Tailwind CSS + Recharts
- **Backend**: FastAPI + SQLite + Pandas
- **AI**: Google Gemini Pro for natural language to SQL conversion

## Data Processing

The system automatically:
- Renames unnamed columns
- Removes empty rows/columns  
- Fills missing values appropriately
- Converts data to SQL-queryable format