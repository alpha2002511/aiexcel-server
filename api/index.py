from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sqlite3
import os
import tempfile
from typing import Dict, Any
import google.generativeai as genai

app = FastAPI()

allowed_origins = os.getenv("ALLOWED_ORIGINS", "https://your-vercel-app.vercel.app").split(",")

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
        self.db_path = os.path.join(tempfile.gettempdir(), "data.db")
        self.init_db()
        
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS user_data (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        
    def clean_excel(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [f"column_{i}" if str(col).startswith('Unnamed') else str(col) for i, col in enumerate(df.columns)]
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(0)
        
        return df
    
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

@app.get("/")
async def root():
    return {"message": "Excel Data Assistant API"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(400, "Only Excel files allowed")
    
    try:
        df = pd.read_excel(file.file, sheet_name=0)
        df = processor.clean_excel(df)
        processor.save_to_db(df, "user_data")
        
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
        conn = sqlite3.connect(processor.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(user_data)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if not columns:
            raise HTTPException(400, "No data uploaded yet")
        
        sample_df = pd.read_sql_query("SELECT * FROM user_data LIMIT 3", conn)
        conn.close()
        
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
        sql_query = sql_query.rstrip(';')
        
        if 'LIMIT' not in sql_query.upper():
            sql_query += ' LIMIT 50'
        
        if not sql_query.upper().startswith('SELECT'):
            raise HTTPException(400, "Invalid query generated")
        
        result_df = processor.query_db(sql_query)
        
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
        raise HTTPException(500, f"Error processing query: {str(e)}")