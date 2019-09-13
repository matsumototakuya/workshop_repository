from flask import Flask, render_template, request, flash
from wtforms import Form, FloatField, IntegerField, SubmitField, validators, ValidationError
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFR
import init_data

# 学習済みモデルを読み込み利用します
def predict(parameters):
    # モデルを読み込み
    model = joblib.load('./rf.pkl')
    params = parameters.reshape(1, -1)
    pred = model.predict(params)
    return pred

def update(parameters, label):
    # データ取得
    data = np.load(file ="data.npy")
    target = np.load(file ="target.npy")
    x = np.append(data,[parameters],axis=0)
    y= np.append(target,[label],axis=0)
    # 訓練データとテストデータに分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    model = RFR(n_jobs=1, random_state=0, n_estimators=5)
    # 学習
    model.fit(x_train, y_train)
    # 学習済みモデル，データセットの保存
    joblib.dump(model, "rf.pkl", compress=True)
    np.save("data", x)
    np.save("target", y)
    score = model.score(x_test, y_test)
    return score

app = Flask(__name__)

# Flaskとwtformsを使い、index.html側で表示させるフォームを構築します。
class DelayForm(Form):
    Time = FloatField("時刻(時)",
                      [validators.InputRequired("この項目は入力必須です"),
                       validators.NumberRange(min=6, max=23)])

    Traffic = FloatField("渋滞状況(%)",
                         [validators.InputRequired("この項目は入力必須です"),
                          validators.NumberRange(min=0, max=100)])

    Passenger = FloatField("乗車率(%)",
                           [validators.InputRequired("この項目は入力必須です"),
                            validators.NumberRange(min=0, max=100)])

    Rainy = FloatField("降水確率(%)",
                       [validators.InputRequired("この項目は入力必須です"),
                        validators.NumberRange(min=0, max=100)])

    # html側で表示するsubmitボタンの表示
    submit = SubmitField("予測")

class LabelForm(Form):
    Label = IntegerField("遅延時間(分)",
                         [validators.InputRequired("この項目は入力必須です"),
                          validators.NumberRange(min=0, max=10)])
    submit = SubmitField("更新")

@app.route('/', methods=['GET', 'POST'])
def predicts():
    form = DelayForm(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            flash("全て入力する必要があります。")
            return render_template('index.html', form=form)
        else:
            Time = float(request.form["Time"])
            Traffic = float(request.form["Traffic"]) / 100
            Passenger = float(request.form["Passenger"]) / 100
            Rainy = float(request.form["Rainy"]) / 100

            x = np.array([Time, Traffic, Passenger, Rainy])
            pred = predict(x)
            np.save("parameters", x)

            return render_template('result.html', pred=round(pred[0],1))
    elif request.method == 'GET':

        return render_template('index.html', form=form)

@app.route('/update/', methods = ['GET', 'POST'])
def updates():
    form = LabelForm(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            flash("全て入力する必要があります。")
            return render_template('update.html', form=form)
        else:
            Label = int(request.form["Label"])
            Parameters = np.load(file ="parameters.npy")
            score = update(Parameters, Label)

            return render_template('update_result.html', score=round(score, 3)*100)
    elif request.method == 'GET':
        return render_template('update.html', form=form)

@app.route('/init/', methods = ['GET', 'POST'])
def inits():
    if request.method == 'POST':
        init_data.init_data()
        return render_template('init_result.html')
    elif request.method == 'GET':
        return render_template('init.html')

@app.route('/list/')
def list():
    data = np.load(file ="data.npy")
    target = np.load(file ="target.npy")
    x = zip(data,target)
    return render_template('list.html', data=x)

if __name__ == "__main__":
    app.run()
