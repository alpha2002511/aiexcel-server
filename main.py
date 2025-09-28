from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sqlite3
import os
import json
from typing import Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Get allowed origins from environment variable
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

class DataProcessor:
    def __init__(self):
        self.db_path = os.getenv("DATABASE_URL", "data.db")
        self.init_db()
        
    def clean_excel(self, df: pd.DataFrame) -> pd.DataFrame:
        # Handle unnamed columns
        df.columns = [f"column_{i}" if str(col).startswith('Unnamed') else str(col) for i, col in enumerate(df.columns)]
        
        # Remove completely empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Fill NaN with appropriate defaults
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(0)
        
        return df
    
    def init_db(self):
        """Initialize database and create tables if needed"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS user_data (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
    
    def save_to_db(self, df: pd.DataFrame, table_name: str):
        conn = sqlite3.connect(self.db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
    
    def query_db(self, query: str) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result

processor = DataProcessor()

@app.get("/test-gemini")
async def test_gemini():
    try:
        response = model.generate_content("Say hello")
        return {"status": "success", "response": response.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(400, "Only Excel files allowed")
    
    try:
        # Read Excel file
        df = pd.read_excel(file.file, sheet_name=0)
        
        # Clean data
        df = processor.clean_excel(df)
        
        # Save to database
        table_name = "user_data"
        processor.save_to_db(df, table_name)
        
        return {
            "message": "File uploaded successfully",
            "columns": list(df.columns),
            "rows": len(df),
            "preview": df.head().to_dict('records')
        }
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")

@app.post("/query")
async def process_query(request: Dict[str, Any]):
    user_question = request.get("question", "")
    
    try:
        # Get table schema and sample data
        conn = sqlite3.connect(processor.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(user_data)")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Get sample data for context
        sample_df = pd.read_sql_query("SELECT * FROM user_data LIMIT 3", conn)
        conn.close()
        
        if not columns:
            raise HTTPException(400, "No data uploaded yet")
        
        # Determine if question needs SQL or descriptive answer
        classification_prompt = f"""
        Question: "{user_question}"
        
        Does this question require:
        A) A SQL query to get specific data (like counts, averages, filtering)
        B) A descriptive explanation about the dataset structure/content
        
        Answer only 'SQL' or 'DESCRIBE'
        """
        
        classification = model.generate_content(classification_prompt)
        query_type = classification.text.strip().upper()
        
        if "DESCRIBE" in query_type or "what is this data" in user_question.lower():
            # Provide descriptive answer
            describe_prompt = f"""
            Dataset with columns: {columns}
            Sample data: {sample_df.to_dict('records')}
            
            Question: "{user_question}"
            
            Provide a clear, helpful description of this dataset and answer the user's question.
            """
            
            response = model.generate_content(describe_prompt)
            return {
                "answer": response.text,
                "data": [],
                "type": "description"
            }
        
        else:
            # Generate and execute SQL query
            sql_prompt = f"""
            Table: user_data
            Columns: {columns}
            
            Convert this question to a SQL SELECT query: "{user_question}"
            
            Rules:
            - Return ONLY the SQL query
            - Use proper column names from the list above
            - Limit results to 50 rows max
            
            SQL:
            """
            
            response = model.generate_content(sql_prompt)
            sql_query = response.text.strip().replace('```sql', '').replace('```', '').strip()
            
            # Clean SQL query - remove semicolon and add LIMIT for non-aggregate queries
            sql_query = sql_query.rstrip(';')
            
            # Only add LIMIT for queries that return multiple rows (not aggregates)
            aggregate_functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP BY']
            is_aggregate = any(func in sql_query.upper() for func in aggregate_functions)
            
            if 'LIMIT' not in sql_query.upper() and not is_aggregate:
                sql_query += ' LIMIT 50'
            
            if not sql_query.upper().startswith('SELECT'):
                raise HTTPException(400, "Invalid query generated")
            
            # Execute query
            result_df = processor.query_db(sql_query)
            
            # Generate natural language summary
            summary_prompt = f"""
            Question: "{user_question}"
            SQL Result: {len(result_df)} rows returned
            
            Provide a brief, natural summary of what was found (1-2 sentences).
            """
            
            summary_response = model.generate_content(summary_prompt)
            
            return {
                "answer": summary_response.text,
                "data": result_df.to_dict('records'),
                "sql": sql_query,
                "type": "query"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(500, f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)