import model_load
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

y_pred = model_load.lin_reg.predict(model_load.train_data[['x1','x2','x3','x4', 'x5']])
R2 = r2_score(model_load.train_data['y'], y_pred)


feature_names = ['x1', 'x2', 'x3', 'x4', 'x5']
equation = [f"{model_load.lin_reg.intercept_:.2f}"]

for coef, feature in zip(model_load.lin_reg.coef_, feature_names):
     if coef >= 0:
          equation.append(f"+ {coef:.2f}{feature}")
     else: equation.append(f"- {abs(coef):.2f}{feature}")
equation_LaTeX = f"$y = {' '.join(equation)}"


def predict_Y(x1,x2,x3,x4,x5):
    
    y_pred = model_load.lin_reg.predict([[x1,x2,x3,x4,x5]])[0]

    plt.figure(figsize=(10,6))
    plt.scatter(model_load.train_data['x1'], model_load.train_data['y'])
    plt.scatter(x1, y_pred, color='red')
    plt.xlabel('x1')
    plt.ylabel('y')

    return y_pred, plt.gcf()