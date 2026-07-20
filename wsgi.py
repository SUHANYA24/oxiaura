"""WSGI entry point for Gunicorn and the Flask dev server.

Run in development:  FLASK_APP=wsgi.py flask run
Run in production:   gunicorn "wsgi:app"
"""

from app import create_app

app = create_app()


if __name__ == "__main__":
    app.run()
