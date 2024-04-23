from flask import Flask
app = Flask(__name__)
@app.route('/')
def first():
    return "<p>这是我的第一个flask程序!</p>"
if __name__ == '__main__':
    app.run()
