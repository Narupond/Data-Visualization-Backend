from flask import Flask, jsonify, request
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# โมเดล
DecisionTree = joblib.load('DecisionTree_model.h5')
GradientBoostedTree = joblib.load('GradientBoostedTree_model.h5')
XGBoost = joblib.load('XGBoost_model.h5')
RandomForest = joblib.load('RandomForest_model.h5')

# รับข้อมูลจากแบบฟอร์ม
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # รับข้อมูลจากแบบฟอร์ม
        data = request.json
        print("-----------------------------")
        print(data)

        # print(DecisionTree)
        # print(GradientBoostedTree)

        # ดำเนินการทำนายด้วยแต่ละโมเดล
        # X = [[123,123,1,22,11,222,33],[123,123,1,22,11,222,33]]
        input_data = [
            data['male'],
            data['female'],
            data['infancy'],
            data['childhood'],
            data['adolescence'],
            data['adulthood'],
            data['elderly'],
        ]
        result_DecisionTree = DecisionTree.predict([input_data])
        result_GradientBoostedTree = GradientBoostedTree.predict([input_data])
        result_XGBoost = XGBoost.predict([input_data])
        result_RandomForest = RandomForest.predict([input_data])

        # สร้าง JSON response
        response = {
            'DecisionTree': [round(i) for i in result_DecisionTree],
            'GradientBoostedTree': [round(i) for i in result_GradientBoostedTree],
            'XGBoost': [round(i) for i in result_XGBoost],
            'RandomForest': [round(i) for i in result_RandomForest],
        }

        print(result_DecisionTree)
        print(result_GradientBoostedTree)
        print(result_XGBoost)
        print(result_RandomForest)

        return jsonify(response)
        # return  jsonify({'test': str(e)}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8888, debug=True)