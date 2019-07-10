"""Application entry point."""
import os
from pathlib import Path
from dotenv import load_dotenv

from app import create_app

if "DYNO" in os.environ:
    # the app is on Heroku
    debug = False
else:
    debug = True
    dotenv_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path)
    
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug, port=port, threaded=True)
