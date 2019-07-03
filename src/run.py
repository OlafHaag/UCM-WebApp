import os
from src.ucmwebapp import app, debug

server = app.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run_server(debug=debug, port=port, threaded=True)
