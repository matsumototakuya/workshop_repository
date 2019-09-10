from flask import Flask, render_template, request, flash
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import numpy as np
from sklearn.externals import joblib


# 学習済みモデルを読み込み利用します
def predict(parameters):
    # ニューラルネットワークのモデルを読み込み
    model = joblib.load('./rf.pkl')
    params = parameters.reshape(1, -1)
    pred = model.predict(params)
    return pred


app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'zJe09C5c3tMf5FnNL09C5d6SAzZoY'


# 公式サイト
# http://wtforms.simplecodes.com/docs/0.6/fields.html
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

            return render_template('result.html', pred=round(pred[0],1))
    elif request.method == 'GET':

        return render_template('index.html', form=form)


if __name__ == "__main__":
    app.run()
