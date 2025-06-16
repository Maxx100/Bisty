import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()
FRONT_PORT = int(os.getenv('FRONT_PORT'))


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=FRONT_PORT, reload=True)
