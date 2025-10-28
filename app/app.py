import gradio as gr

# мои модули
import model_load
import work_with_model


description = f"""
    ## Модель линейной регрессии

    Качество модели (R^2 на тестовой выборке): {work_with_model.R2}

    {work_with_model.equation_LaTeX}
    """

# функция для работы с моделью выносится в отдельный модуль.
# с помощью своего обработчика нажатия на кнопку submit я могу вносить дополнительные обработчики исключений перед вызовом функции. 
def on_submit(x1,x2,x3,x4,x5):
     return work_with_model.predict_Y(x1,x2,x3,x4,x5)


demo = gr.Interface(
   fn = on_submit,
   inputs=["number", "number", "number", "number", "number"],
   outputs=[gr.Number(label="y_prediction"), gr.Plot(label="scatter диаграмма")],
   description = description
)


demo.launch()