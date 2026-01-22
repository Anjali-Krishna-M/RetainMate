from app import create_app

app = create_app()

if __name__ == '__main__':
    # Debug=True means the website updates automatically when you save code
    app.run(debug=True)