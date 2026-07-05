# ImageLab

Streamlit-интерфейс для параллельного анализа концентрации талька, поиска тёмных включений внутри сульфида и классификации руды моделью ResNet18.

## Запуск

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Откройте адрес, который Streamlit покажет в терминале (обычно `http://localhost:8501`).

## Алгоритмы обработки

Вычислительная логика вынесена в `image_processing.py`. Функции `process_talc_concentration` и `process_sulfide_inclusions` запускаются параллельно из `app.py`.

Классификатор находится в `ore_classifier.py`, а безопасный `state_dict` модели — в `models/resnet18_epoch_10_state_dict.pt`. ResNet18 запускается автоматически при доле талька менее 10% либо принудительно кнопкой в интерфейсе. Для классификации исходное изображение должно быть не меньше 1024×1024 пикселей.

Вспомогательная сегментация SAM находится в `sam_predictor.py`, веса — в `models/best_model_b_v2.pth`. SAM загружается и запускается только по кнопке пользователя; результат отображается под классификацией ResNet18 и добавляется в PDF-отчёт.
