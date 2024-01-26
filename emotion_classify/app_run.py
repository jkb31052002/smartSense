import subprocess
import threading

def run_fastapi():
    subprocess.run(["uvicorn", "backend.main:app", "--reload"])

def run_streamlit():
    subprocess.run(["streamlit", "run", "frontend/frontend_streamlit.py"])

if __name__ == "__main__":
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()

    run_streamlit()
