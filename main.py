from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sqlite3
import os
import json
from typing import Dict, Any, List
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

def _get_generation_config():
    return {
        "temperature": float(os.getenv("GEN_TEMPERATURE", "0.2")),
        "top_p": float(os.getenv("GEN_TOP_P", "0.9")),
        "top_k": int(os.getenv("GEN_TOP_K", "40")),
        "max_output_tokens": int(os.getenv("GEN_MAX_TOKENS", "512")),
    }

SAFE_SQL_PREFIX = "SELECT"
MAX_ROWS = int(os.getenv("MAX_ROWS", "50"))


def _build_classification_prompt(user_question: str) -> str:
    return f"""
You are a helpful analyst. Classify the user's request strictly as one of two categories.

Question: "{user_question}"

Return EXACTLY one token: SQL or DESCRIBE.
- Return SQL if the user is asking to compute values, filter rows, aggregate, sort, or otherwise needs data retrieved using a SELECT statement.
- Return DESCRIBE if the user is asking about the structure, columns, ranges, freshness, or wants a narrative/overview insight.
""".strip()


def _build_sql_prompt(columns: List[str], user_question: str) -> str:
    column_list = ", ".join(columns)
    return f"""
You convert natural language questions into a single SQLite SELECT statement over one table called user_data.

Columns: {column_list}

Rules:
- Output ONLY the SQL, no commentary, no markdown fences.
- Use ONLY these columns and ONLY the table user_data.
- Do NOT write DDL/DML (no CREATE/UPDATE/DELETE/INSERT/DROP/ATTACH/PRAGMA).
- If aggregating, include GROUP BY when needed.
- Prefer explicit column names over * where possible.
- Always limit the result to {MAX_ROWS} rows if it could return many rows.
- Use double quotes around string literals only if required by SQLite; otherwise single quotes are fine.
- If the question is ambiguous, make a reasonable assumption and proceed.

Question: "{user_question}"
SQL:
""".strip()


def _is_sql_safe(sql_query: str, columns: List[str]) -> bool:
    sql_upper = sql_query.upper()
    if not sql_upper.startswith(SAFE_SQL_PREFIX):
        return False
    forbidden = [
        ";", "--", "/*", "*/", " PRAGMA ", " ATTACH ", " DETACH ",
        " UPDATE ", " INSERT ", " DELETE ", " DROP ", " ALTER ", " CREATE ", " VACUUM ",
    ]
    if any(f in sql_upper for f in forbidden):
        return False
    # Ensure only known columns or '*' and basic SQL keywords are present
    # Basic heuristic: if identifiers not in allowed set, reject
    allowed_tokens = {c.upper() for c in columns}
    allowed_tokens.update({"SELECT","FROM","WHERE","AND","OR","GROUP","BY","ORDER","LIMIT","ASC","DESC","COUNT","SUM","AVG","MIN","MAX","AS","LIKE","IN","NOT","BETWEEN","HAVING","DISTINCT","CASE","WHEN","THEN","ELSE","END","IS","NULL","ON" ,"INNER","LEFT","JOIN"})
    # Very light check: split on non-alphanum/underscore
    import re
    tokens = [t for t in re.split(r"[^A-Z0-9_]+", sql_upper) if t]
    unknown_identifiers = [t for t in tokens if t.isalpha() and t not in allowed_tokens and t != "USER_DATA" and not t.isnumeric()]
    # Allow function names
    allowed_funcs = {"COUNT","SUM","AVG","MIN","MAX"}
    unknown_identifiers = [t for t in unknown_identifiers if t not in allowed_funcs]
    return len(unknown_identifiers) == 0


def _ensure_limit(sql_query: str, is_aggregate: bool) -> str:
    if "LIMIT" in sql_query.upper() or is_aggregate:
        return sql_query
    return f"{sql_query} LIMIT {MAX_ROWS}"


def _summarize_result(user_question: str, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No rows matched the criteria."
    # Build a compact, useful summary
    max_preview_rows = 3
    num_rows = len(df)
    num_cols = len(df.columns)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    parts = [
        f"Returned {num_rows} rows and {num_cols} columns.",
    ]
    if numeric_cols:
        desc = df[numeric_cols].describe().to_dict()
        # keep only mean for brevity
        means = {k: v.get("mean") for k, v in desc.items() if isinstance(v, dict) and "mean" in v}
        if means:
            parts.append(f"Key averages: {json.dumps(means, default=float)}")
    parts.append(f"Preview: {df.head(max_preview_rows).to_dict('records')}")
    return " ".join(parts)

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

@app.get("/")
async def root():
    return {"message": "Excel Data Assistant API"}

@app.get("/health")
async def health():
    return {"status": "ok"}

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
        classification_prompt = _build_classification_prompt(user_question)
        
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
            sql_prompt = _build_sql_prompt(columns, user_question)
            
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