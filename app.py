from flask import Flask,render_template,request,jsonify
import pickle
#intilization of flask

app = Flask(__name__)
model = pickle.load(open("sentiment-classification-xg-boost-model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/predict",methods=['POST'])
def predict():
    user = request.form['userName']
    user = user.lower()
    prediction = model.predict(user)
    return render_template("index.html",prediction_text=prediction)

if __name__=="__main__":
    app.run(debug=True)
